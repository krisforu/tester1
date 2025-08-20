# -*- coding: utf-8 -*-
"""
LLM-only 360° Equity Report Generator
- Input: data/companies.xlsx or data/companies.csv with columns: NSE_Symbol, BSE_Symbol, BSE_Code, Company
- Output: CompanyReports/<IDENTIFIER>.pdf
- No external data APIs. OpenAI only.
- The model is asked to use the latest public info it can access; if unavailable, it should mark numbers as N/A and proceed.
"""

import os, json, pathlib, re, traceback
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from tqdm import tqdm

from openai import OpenAI

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ---------- CONFIG ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "high")
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "6500"))

INPUT_PATH = os.getenv("INPUT_PATH", "data/companies.xlsx")
OUTPUT_DIR = pathlib.Path(os.getenv("OUTPUT_DIR", "CompanyReports"))
OUTPUT_DIR.mkdir(exist_ok=True)

# One and only required key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MASTER_PROMPT = """You are a world-class equity research analyst, technical analyst, and financial news summarizer combined.
You will produce a complete 360° analysis for one Indian stock (NSE/BSE).

IMPORTANT INSTRUCTIONS
- Use the latest public information you have access to. If any figure is unknown, unverifiable, or not recent enough, write **N/A** and continue.
- Prefer trailing 12 months (TTM) for fundamentals. Include dividend yield, promoter holding, and institutional holding when known.
- Technicals: provide DAILY and WEEKLY views. If you can’t compute exact indicators, provide reasonable ranges and a clear disclaimer.
- Keep the language tight and professional. Use small tables where useful.
- End with exactly 5 short “Key Takeaways” bullets.

STRUCTURE (exact headings):

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

Finish with:  • Key Takeaways (5 bullets)
If you make assumptions, clearly label them.
"""

# ---------- UTILS ----------
def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:90]

def resolve_identifier(row: dict) -> Optional[str]:
    for field in ["NSE_Symbol", "BSE_Symbol", "BSE_Code", "Company"]:
        val = str(row.get(field, "")).strip()
        if val and val.lower() != "nan":
            return val
    return None

def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Muted", parent=styles["BodyText"], textColor=colors.HexColor("#6c757d")))
    return styles

def build_pdf(identifier: str, text: str):
    styles = make_styles()
    story = []
    now = datetime.now().strftime("%d %b %Y, %H:%M")

    story.append(Paragraph(f"<b>{identifier} — 360° Equity Report</b>", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<font color='#6c757d'>Generated: {now}</font>", styles["Muted"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("AI Analysis", styles["Heading2"]))
    story.append(Spacer(1, 6))

    for para in text.split("\n\n"):
        story.append(Paragraph(para.replace("\n","<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 6))

    pdf_path = OUTPUT_DIR / f"{safe_filename(identifier)}.pdf"
    doc = SimpleDocTemplate(str(pdf_path),
                            pagesize=A4,
                            leftMargin=36, rightMargin=36,
                            topMargin=36, bottomMargin=36)
    doc.build(story)

@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(6),
       retry=retry_if_exception_type(Exception))
def generate_report(identifier: str) -> Tuple[str, dict]:
    prompt = f"""{MASTER_PROMPT}

Now analyze this stock (NSE/BSE India): **{identifier}**
Make it investment-grade and pragmatic. If you aren’t fully sure of a metric, write N/A with a brief note."""
    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        reasoning={"effort": REASONING_EFFORT},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    usage = {
        "input_tokens": getattr(resp.usage, "input_tokens", 0) if hasattr(resp, "usage") else 0,
        "output_tokens": getattr(resp.usage, "output_tokens", 0) if hasattr(resp, "usage") else 0,
        "total_tokens": getattr(resp.usage, "total_tokens", 0) if hasattr(resp, "usage") else 0,
    }
    return resp.output_text, usage

def read_companies(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if p.suffix.lower() in [".xlsx",".xls"]:
        return pd.read_excel(p)
    if p.suffix.lower()==".csv":
        return pd.read_csv(p)
    raise ValueError("Use .xlsx/.xls or .csv")

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")

    # token log
    log_file = OUTPUT_DIR / "usage_log.csv"
    if not log_file.exists():
        log_file.write_text("Identifier,InputTokens,OutputTokens,TotalTokens\n", encoding="utf-8")

    df = read_companies(INPUT_PATH)

    with tqdm(total=len(df), desc="Reports") as pbar:
        for _, row in df.iterrows():
            identifier = resolve_identifier(row)
            if not identifier:
                pbar.update(1); continue

            outpdf = OUTPUT_DIR / f"{safe_filename(identifier)}.pdf"
            if outpdf.exists():
                pbar.update(1); continue

            try:
                text, usage = generate_report(identifier)
                build_pdf(identifier, text)
                with open(log_file, "a", encoding="utf-8") as lf:
                    lf.write(f"{identifier},{usage['input_tokens']},{usage['output_tokens']},{usage['total_tokens']}\n")
            except Exception:
                errp = OUTPUT_DIR / f"{safe_filename(identifier)}_ERROR.txt"
                with open(errp, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
            pbar.update(1)

if __name__ == "__main__":
    main()
