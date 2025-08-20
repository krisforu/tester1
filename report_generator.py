# -*- coding: utf-8 -*-
"""
Batch 360° Equity Report Generator (India)
- Input: data/companies.xlsx or data/companies.csv (has columns: NSE_Symbol, BSE_Symbol, BSE_Code, Company)
- Output: CompanyReports/<IDENTIFIER>.pdf (+ charts) + usage_log.csv
- Runs fine on GitHub Actions (no local installs needed)
- No yfinance; prefers NSEpy (if not blocked) else AlphaVantage fallback; Screener scrape for fundamentals.
"""

import os, json, pathlib, re, math, io, traceback
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import feedparser
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from openai import OpenAI

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# PDF
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm

# Optional NSE history
try:
    from nsepy import get_history as nse_get_history
    NSEPY_OK = True
except Exception:
    NSEPY_OK = False

# ------------------ CONFIG ------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "high")
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "6000"))

INPUT_PATH = os.getenv("INPUT_PATH", "data/companies.xlsx")  # or data/companies.csv
OUTPUT_DIR = pathlib.Path(os.getenv("OUTPUT_DIR", "CompanyReports"))
OUTPUT_DIR.mkdir(exist_ok=True)

AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()  # optional
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()      # optional

THEME = {
    "brand": "#0b6efd",
    "accent": "#0d6efd",
    "muted": "#6c757d",
    "table_header_bg": "#e9f2ff",
}

MASTER_PROMPT = """You are a world-class equity research analyst, technical analyst, and financial news summarizer combined.
Use the JSON context below (facts & numbers) to produce a 360° report.

Sections (in order):
1) FUNDAMENTAL ANALYSIS
- Company Overview
- Financial Health (TTM: Revenue Growth, Profit, EPS, Margins, Debt/Equity, Cash Flow)
- Valuation vs Competitors
- Growth Potential
- Risks
- Recent Catalysts
- Dividend Yield, Promoter Holding, Institutional Holding
- Verdict (Bull Case, Bear Case, Short & Long Term outlook, Buffett view)

2) TECHNICAL ANALYSIS
- Current Price, % change
- 52-week High/Low
- Key Support & Resistance
- Moving Averages
- RSI, MACD, Stochastic (Daily & Weekly)
- Trend & Chart Patterns
- Trading Plan (Entry, Target, Stop Loss)

3) NEWS & SENTIMENT
- Market-wide news
- Latest 5–10 news specific to the stock (1–2 lines each + sentiment tag)
- Recurring themes
- Sentiment Score (0–10)

4) PEER & SCREENER INSIGHT
- Compare with top 3 competitors (table: valuation, growth, profitability, market cap)
- Suggest 2–3 alternative strong/undervalued stocks

Finish with 5 short “Key Takeaways”.
Use only provided JSON for figures. If missing, say so plainly.
"""

UA = {"User-Agent": "Mozilla/5.0 (+https://github.com/your-org/your-repo)"}
VADER = SentimentIntensityAnalyzer()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------ UTILS ------------------
def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:90]

def to_float(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None

def as_pct(x, d=1):
    return f"{x:.{d}f}%" if x is not None else "—"

def resolve_from_row(row: dict) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    nse = (str(row.get("NSE_Symbol", "")).strip() or None)
    bse = (str(row.get("BSE_Symbol", "")).strip() or None)
    bse_code = (str(row.get("BSE_Code", "")).strip() or None)
    company = (str(row.get("Company", "")).strip() or None)
    for v in ("nse","bse","bse_code","company"):
        pass
    return nse, bse, bse_code, company

# Optional manual overrides (common mappings)
OVERRIDES = {
    # "HITACHI ENERGY": {"nse": "POWERINDIA"},
}

# ------------------ Screener scrape ------------------
def screener_slug_guess(nse_symbol: Optional[str], company: Optional[str]) -> Optional[str]:
    cands = []
    if nse_symbol:
        cands += [nse_symbol.upper(), nse_symbol.lower(), nse_symbol.title()]
    if company:
        base = re.sub(r"(limited|ltd\.?|inc|india|company)", "", company, flags=re.I)
        tokens = re.sub(r"[^A-Za-z0-9 ]+", " ", base).split()
        if tokens:
            cands += ["-".join(t.lower() for t in tokens[:2])]
            cands += ["-".join(t.lower() for t in tokens[:3])]
            cands += [tokens[0].lower()]
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            out.append(c); seen.add(c)
    for c in out:
        url = f"https://www.screener.in/company/{c}/consolidated/"
        try:
            r = requests.get(url, headers=UA, timeout=15)
            if r.status_code == 200 and "Company not found" not in r.text:
                return c
        except Exception:
            continue
    return None

def parse_screener(slug: str) -> Dict[str, Any]:
    url = f"https://www.screener.in/company/{slug}/consolidated/"
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    out = {
        "company_name": (soup.select_one("h1 a").text.strip() if soup.select_one("h1 a") else None),
        "market_cap_cr": None, "pe": None, "roe_pct": None, "roce_pct": None,
        "de_ratio": None, "div_yield_pct": None, "ttm_sales_cr": None,
        "ttm_profit_cr": None, "ttm_op_margin_pct": None,
        "promoter_holding_pct": None, "fii_holding_pct": None, "dii_holding_pct": None,
        "url": url
    }

    for li in soup.select("li.flex.flex-space-between"):
        lab = li.select_one("span.name"); val = li.select_one("span.value")
        if not lab or not val: continue
        k = lab.text.strip().lower(); v = val.text.strip().replace(",", "")
        if "market cap" in k:
            m = re.search(r"([\d.]+)\s*cr", v, re.I)
            out["market_cap_cr"] = float(m.group(1)) if m else to_float(v)
        elif "stock p/e" in k:
            out["pe"] = to_float(v)
        elif "dividend yield" in k:
            m = re.search(r"([\d.]+)\s*%", v); out["div_yield_pct"] = float(m.group(1)) if m else None
        elif "roce" in k:
            m = re.search(r"([\d.]+)", v); out["roce_pct"] = float(m.group(1)) if m else None
        elif "roe" in k:
            m = re.search(r"([\d.]+)", v); out["roe_pct"] = float(m.group(1)) if m else None

    # Ownership
    for row in soup.select("table.data-table tr"):
        cells = [c.get_text(strip=True) for c in row.select("td, th")]
        if len(cells) < 2: continue
        title = cells[0].lower()
        if "promoters" in title:
            nums = re.findall(r"([\d.]+)\s*%", " ".join(cells[1:])); 
            if nums: out["promoter_holding_pct"] = float(nums[-1])
        if "fii" in title:
            nums = re.findall(r"([\d.]+)\s*%", " ".join(cells[1:])); 
            if nums: out["fii_holding_pct"] = float(nums[-1])
        if "dii" in title:
            nums = re.findall(r"([\d.]+)\s*%", " ".join(cells[1:])); 
            if nums: out["dii_holding_pct"] = float(nums[-1])

    # TTM hints (heuristic)
    for card in soup.select("div.card"):
        head = card.select_one("h2")
        if not head: continue
        h = head.get_text(strip=True).lower()
        if "ratios" in h:
            for tr in card.select("table tr"):
                tds = [c.get_text(" ", strip=True) for c in tr.select("td, th")]
                if len(tds) < 2: continue
                key = tds[0].lower()
                if "debt to equity" in key: out["de_ratio"] = to_float(tds[-1])
                if "operating profit margin" in key or "opm" in key:
                    out["ttm_op_margin_pct"] = to_float(tds[-1])
        if "profit & loss" in h or "profit and loss" in h:
            for tr in card.select("table tr"):
                tds = [c.get_text(" ", strip=True) for c in tr.select("td, th")]
                if len(tds) < 3: continue
                if "ttm" in tds[0].lower():
                    nums = [to_float(x.replace(" cr","").replace("%","")) for x in tds[1:]]
                    if len(nums) >= 1: out["ttm_sales_cr"] = nums[0]
                    if len(nums) >= 9: out["ttm_profit_cr"] = nums[8]
    return out

# ------------------ Price history ------------------
def fetch_history_nse(symbol: str, years:int=3) -> Optional[pd.DataFrame]:
    if not NSEPY_OK: return None
    try:
        end = date.today(); start = end - timedelta(days=365*years + 14)
        df = nse_get_history(symbol=symbol.upper(), start=start, end=end)
        if df is None or df.empty: return None
        df = df.reset_index()
        df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
        return df[["date","open","high","low","close","volume"]]
    except Exception:
        return None

def fetch_history_av(symbol: str, primary="NSE") -> Optional[pd.DataFrame]:
    if not AV_KEY: return None
    syms = [f"{symbol}.NSE", f"{symbol}.BSE"] if primary.upper()=="NSE" else [f"{symbol}.BSE", f"{symbol}.NSE"]
    for sym in syms:
        try:
            r = requests.get("https://www.alphavantage.co/query",
                             params={"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":sym,"apikey":AV_KEY,"outputsize":"full"},
                             timeout=25)
            js = r.json()
            if "Time Series (Daily)" not in js: continue
            rows=[]
            for d,vals in js["Time Series (Daily)"].items():
                rows.append({"date": pd.to_datetime(d).date(),
                             "open": float(vals["1. open"]),
                             "high": float(vals["2. high"]),
                             "low":  float(vals["3. low"]),
                             "close":float(vals["4. close"]),
                             "volume":float(vals["6. volume"])})
            return pd.DataFrame(rows).sort_values("date")
        except Exception:
            continue
    return None

def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    out = {"daily":{}, "weekly":{}}
    if df is None or df.empty: return out
    df = df.sort_values("date").copy()

    for n in [20,50,200]:
        df[f"ma{n}"] = df["close"].rolling(n).mean()

    # RSI14
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    df["rsi14"] = 100 - (100/(1+rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"], df["macd_signal"] = macd, signal

    # Stoch
    ll = df["low"].rolling(14).min()
    hh = df["high"].rolling(14).max()
    k = (df["close"]-ll)/(hh-ll)*100
    d = k.rolling(3).mean()
    df["stoch_k"], df["stoch_d"] = k, d

    last = df.iloc[-1]
    last_date = last["date"]
    df52 = df[df["date"] >= (last_date - timedelta(days=365))]
    out["daily"] = {
        "price": float(last["close"]),
        "ma20": float(last.get("ma20", np.nan)) if not math.isnan(last.get("ma20", np.nan)) else None,
        "ma50": float(last.get("ma50", np.nan)) if not math.isnan(last.get("ma50", np.nan)) else None,
        "ma200":float(last.get("ma200", np.nan))if not math.isnan(last.get("ma200", np.nan)) else None,
        "rsi14": float(last.get("rsi14", np.nan)) if not math.isnan(last.get("rsi14", np.nan)) else None,
        "macd": float(last.get("macd", 0.0)), "macd_signal": float(last.get("macd_signal", 0.0)),
        "stoch_k": float(last.get("stoch_k", np.nan)) if not math.isnan(last.get("stoch_k", np.nan)) else None,
        "stoch_d": float(last.get("stoch_d", np.nan)) if not math.isnan(last.get("stoch_d", np.nan)) else None,
        "high_52w": float(df52["close"].max()), "low_52w": float(df52["close"].min())
    }

    # Weekly aggregates
    w = df.copy(); w["dt"] = pd.to_datetime(w["date"])
    w = w.set_index("dt").resample("W-FRI").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
    for n in [10,30]: w[f"ma{n}"] = w["close"].rolling(n).mean()
    dlt = w["close"].diff()
    gain = dlt.clip(lower=0).rolling(14).mean()
    loss = (-dlt.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    w["rsi14"] = 100 - (100/(1+rs))
    ema12 = w["close"].ewm(span=12, adjust=False).mean()
    ema26 = w["close"].ewm(span=26, adjust=False).mean()
    w["macd"] = ema12 - ema26
    w["macd_signal"] = w["macd"].ewm(span=9, adjust=False).mean()
    low14 = w["low"].rolling(14).min(); high14 = w["high"].rolling(14).max()
    w["stoch_k"] = (w["close"]-low14)/(high14-low14)*100
    w["stoch_d"] = w["stoch_k"].rolling(3).mean()

    lastw = w.iloc[-1]
    out["weekly"] = {
        "ma10": float(lastw.get("ma10", np.nan)) if not math.isnan(lastw.get("ma10", np.nan)) else None,
        "ma30": float(lastw.get("ma30", np.nan)) if not math.isnan(lastw.get("ma30", np.nan)) else None,
        "rsi14": float(lastw.get("rsi14", np.nan)) if not math.isnan(lastw.get("rsi14", np.nan)) else None,
        "macd": float(lastw.get("macd", 0.0)), "macd_signal": float(lastw.get("macd_signal", 0.0)),
        "stoch_k": float(lastw.get("stoch_k", np.nan)) if not math.isnan(lastw.get("stoch_k", np.nan)) else None,
        "stoch_d": float(lastw.get("stoch_d", np.nan)) if not math.isnan(lastw.get("stoch_d", np.nan)) else None,
    }
    return out

def draw_chart(df: pd.DataFrame, title: str, outpng: pathlib.Path):
    if df is None or df.empty: return
    fig = plt.figure(figsize=(7.5, 3.8), dpi=150)
    plt.plot(df["date"], df["close"], label="Close", linewidth=1.2)
    for n, lab in [(20,"MA20"), (50,"MA50"), (200,"MA200")]:
        plt.plot(df["date"], df["close"].rolling(n).mean(), label=lab, linewidth=1.0, alpha=0.9)
    plt.title(title); plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend(loc="upper left", fontsize=7); plt.grid(alpha=0.25)
    plt.tight_layout(); fig.savefig(outpng); plt.close(fig)

# ------------------ News ------------------
def google_news(query: str, k=10):
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:k]:
        title = e.title
        link = e.link
        when = None
        if hasattr(e, "published_parsed") and e.published_parsed:
            when = datetime(*e.published_parsed[:6])
        score = VADER.polarity_scores(title)["compound"]
        sent = "Positive" if score>=0.2 else "Negative" if score<=-0.2 else "Neutral"
        items.append({"title": title, "url": link, "published": str(when) if when else None, "sentiment": sent, "score": score})
    return items

def company_news(company: str, sym_hint: Optional[str]) -> List[Dict[str, Any]]:
    q = company or (sym_hint or "")
    return google_news(q, k=10)

# ------------------ LLM ------------------
@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(6),
       retry=retry_if_exception_type(Exception))
def call_llm(identifier: str, ctx: Dict[str, Any]) -> Tuple[str, Any]:
    ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)
    prompt = f"""<DATA JSON>
{ctx_json}
</DATA JSON>

{MASTER_PROMPT}
"""
    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        reasoning={"effort": REASONING_EFFORT},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return resp.output_text, getattr(resp, "usage", None)

# ------------------ PDF ------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self._saved = []
    def showPage(self):
        self._saved.append(dict(self.__dict__)); super().showPage()
    def save(self):
        pages = len(self._saved)
        for s in self._saved:
            self.__dict__.update(s)
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor(THEME["muted"]))
            w,h=A4
            self.drawRightString(w-1.5*cm, 1.0*cm, f"Page {self._pageNumber} of {pages}")
            super().showPage()
        super().save()

def styles():
    st = getSampleStyleSheet()
    st.add(ParagraphStyle(name="Muted", parent=st["BodyText"], textColor=colors.HexColor(THEME["muted"])))
    st.add(ParagraphStyle(name="H2Brand", parent=st["Heading2"], textColor=colors.HexColor(THEME["accent"])))
    return st

def build_pdf(identifier: str, ctx: Dict[str, Any], llm_text: str, chart: Optional[pathlib.Path]):
    st = styles()
    story = []
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    res = ctx.get("resolved", {})
    symline = " / ".join([x for x in [res.get("nse_symbol"), res.get("bse_symbol"), res.get("bse_code")] if x])

    story.append(Paragraph(f"<b>{identifier} — 360° Equity Report</b>", st["Title"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<font color='{THEME['muted']}'>Generated: {now} • Symbols: {symline}</font>", st["Muted"]))
    story.append(Spacer(1, 8))

    # quick facts table
    f = ctx.get("fundamentals", {})
    t = ctx.get("technicals", {})
    facts = []
    if f.get("market_cap_cr"): facts.append(["Mkt Cap (₹ Cr)", f"{f['market_cap_cr']:.1f}"])
    if f.get("pe") is not None: facts.append(["P/E (TTM)", f"{f['pe']:.1f}"])
    if f.get("roce_pct") is not None: facts.append(["ROCE", as_pct(f['roce_pct'])])
    if f.get("roe_pct")  is not None: facts.append(["ROE",  as_pct(f['roe_pct'])])
    if f.get("de_ratio") is not None: facts.append(["Debt/Equity", f"{f['de_ratio']:.2f}"])
    if f.get("div_yield_pct") is not None: facts.append(["Dividend Yield", as_pct(f['div_yield_pct'])])
    if t.get("daily", {}).get("high_52w") is not None:
        facts.append(["52W High/Low", f"{t['daily']['high_52w']:.2f} / {t['daily']['low_52w']:.2f}"])
    if facts:
        tbl = Table([["Metric","Value"]] + facts, colWidths=[6*cm, 10*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(THEME["table_header_bg"])),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("LINEABOVE",(0,0),(-1,0),0.25,colors.HexColor(THEME["brand"])),
            ("LINEBELOW",(0,0),(-1,0),0.25,colors.HexColor(THEME["brand"])),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke, colors.Color(1,1,1)]),
            ("LEFTPADDING",(0,0),(-1,-1),6), ("RIGHTPADDING",(0,0),(-1,-1),6),
        ]))
        story.append(tbl); story.append(Spacer(1, 8))

    if chart and chart.exists():
        story.append(Paragraph("Price (3Y Daily) — MA20/50/200", st["H2Brand"]))
        story.append(Spacer(1, 4))
        story.append(Image(str(chart), width=17*cm, height=8*cm))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Full Analysis", st["H2Brand"]))
    story.append(Spacer(1, 4))
    for para in llm_text.split("\n\n"):
        story.append(Paragraph(para.replace("\n","<br/>"), st["BodyText"]))
        story.append(Spacer(1, 6))

    pdf_path = OUTPUT_DIR / f"{safe_filename(identifier)}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            leftMargin=36, rightMargin=36,
                            topMargin=48, bottomMargin=36)
    doc.build(story, canvasmaker=NumberedCanvas)

# ------------------ Context build ------------------
def build_context(nse: Optional[str], bse: Optional[str], bse_code: Optional[str], company: Optional[str]) -> Dict[str, Any]:
    # apply overrides
    for k,v in OVERRIDES.items():
        if company and company.strip().upper()==k.upper():
            nse = v.get("nse", nse); bse = v.get("bse", bse); bse_code = v.get("bse_code", bse_code)

    # fundamentals
    fundamentals = {}
    slug = screener_slug_guess(nse, company)
    if slug:
        try: fundamentals = parse_screener(slug)
        except Exception: fundamentals = {}

    # price/technicals
    hist = None
    if nse and NSEPY_OK:
        hist = fetch_history_nse(nse, years=3)
    if hist is None and nse:
        hist = fetch_history_av(nse, primary="NSE")
    if hist is None and bse:
        hist = fetch_history_av(bse, primary="BSE")

    technicals = compute_indicators(hist) if hist is not None else {}

    # news
    news = company_news(company or (nse or bse or ""), sym_hint=nse or bse)

    return {
        "resolved": {"nse_symbol": nse, "bse_symbol": bse, "bse_code": bse_code, "company": company},
        "fundamentals": fundamentals,
        "technicals": technicals,
        "news": news,
        "source_note": {
            "prices": "NSEpy (if available) or AlphaVantage fallback",
            "fundamentals": fundamentals.get("url"),
            "news": "Google News RSS (headline sentiment via VADER)"
        }
    }

# ------------------ MAIN ------------------
def read_companies(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError("Use .xlsx/.xls or .csv for input list.")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required (set it as a GitHub Secret).")

    # usage log
    log_file = OUTPUT_DIR / "usage_log.csv"
    if not log_file.exists():
        log_file.write_text("Identifier,InputTokens,OutputTokens,TotalTokens,CostUSD\n", encoding="utf-8")

    df = read_companies(INPUT_PATH)
    with tqdm(total=len(df), desc="Building reports") as pbar:
        for _, row in df.iterrows():
            nse, bse, bse_code, company = resolve_from_row(row)
            identifier = nse or bse or bse_code or company
            if not identifier:
                pbar.update(1); continue

            outpdf = OUTPUT_DIR / f"{safe_filename(identifier)}.pdf"
            if outpdf.exists():
                pbar.update(1); continue

            try:
                ctx = build_context(nse, bse, bse_code, company)
                # LLM
                text, usage = call_llm(identifier, ctx)

                # chart
                chart_path = None
                # re-fetch for chart to keep function boundaries simple
                hist = None
                if nse and NSEPY_OK: hist = fetch_history_nse(nse, years=3)
                if hist is None and nse: hist = fetch_history_av(nse, primary="NSE")
                if hist is None and bse: hist = fetch_history_av(bse, primary="BSE")
                if hist is not None and not hist.empty:
                    chart_path = OUTPUT_DIR / f"{safe_filename(identifier)}_chart.png"
                    draw_chart(hist, f"{identifier} — 3Y Daily", chart_path)

                # PDF
                build_pdf(identifier, ctx, text, chart_path)

                # Token log (rough pricing example; adjust to your rates)
                if usage:
                    it = getattr(usage, "input_tokens", 0) or 0
                    ot = getattr(usage, "output_tokens", 0) or 0
                    tt = getattr(usage, "total_tokens", it+ot)
                    cost = (it/1e6 * 1.25) + (ot/1e6 * 10.0)
                    with open(log_file, "a", encoding="utf-8") as lf:
                        lf.write(f"{identifier},{it},{ot},{tt},{cost:.6f}\n")

            except Exception as e:
                errp = OUTPUT_DIR / f"{safe_filename(identifier)}_ERROR.txt"
                with open(errp, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())

            pbar.update(1)

if __name__ == "__main__":
    main()

