# scripts/run_batches.py
import os, re, json, time, pathlib
from typing import List, Dict, Any
import pandas as pd

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from openai import OpenAI
from openai import APIStatusError

try:
    from pypdf import PdfMerger
    HAVE_PYPDF = True
except Exception:
    HAVE_PYPDF = False

MODEL = os.getenv("RESP_MODEL", "gpt-5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
SDK_MAX_RETRIES = int(os.getenv("SDK_MAX_RETRIES", "0"))  # no SDK retries to avoid extra spend
STOP_IF_EMPTY = os.getenv("STOP_IF_EMPTY", "1") not in ("0", "false", "False")

OUTDIR = pathlib.Path("build"); OUTDIR.mkdir(parents=True, exist_ok=True)
MASTER_CSV = OUTDIR / "AllCompanies_Quarterly.csv"
MASTER_PDF = OUTDIR / "AllCompanies_Report.pdf"

def _pick(df: pd.DataFrame, *candidates: str) -> str | None:
    low = {str(c).strip().lower(): c for c in df.columns}
    for n in candidates:
        if n.lower() in low:
            return low[n.lower()]
    return None

def _find_excel() -> str:
    for p in ("data/companies.xlsx", "data/companies10.xlsx"):
        if os.path.exists(p): return p
    raise FileNotFoundError("Expected data/companies.xlsx (or data/companies10.xlsx).")

def _clean_bse_code(v) -> str:
    if pd.isna(v): return ""
    s = str(v).strip()
    try: return str(int(float(s)))
    except Exception: return s

def load_companies_xlsx() -> List[Dict[str, str]]:
    path = _find_excel()
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    c_name = _pick(df, "Company", "Name", "Company Name")
    c_nse  = _pick(df, "NSE_Symbol", "NSE Symbol", "NSE")
    c_bses = _pick(df, "BSE_Symbol", "BSE Symbol", "BSE")
    c_bsec = _pick(df, "BSE_Code", "BSE Code", "Code")
    if not c_name:
        raise ValueError("Excel must contain a company name column (e.g., 'Company').")
    rows: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        rows.append({
            "name": str(r[c_name]).strip(),
            "nse_symbol": "" if not c_nse or pd.isna(r.get(c_nse)) else str(r[c_nse]).strip(),
            "bse_symbol": "" if not c_bses or pd.isna(r.get(c_bses)) else str(r[c_bses]).strip(),
            "bse_code": "" if not c_bsec or pd.isna(r.get(c_bsec)) else _clean_bse_code(r[c_bsec]),
        })
    return rows

def chunk(lst: List[Dict[str, str]], size: int):
    for i in range(0, len(lst), size):
        yield (i // size) + 1, lst[i:i+size]

# ---------- LLM call with JSON hardening ----------
def _json_hard_prompt(companies: List[Dict[str, str]]) -> str:
    return (
        "You are a data extraction worker. Using Screener.in where possible, return ONLY valid JSON.\n"
        "Do not add any prose or markdown, just JSON.\n"
        "Schema:\n"
        "{\n"
        '  "companies": [\n'
        "    {\n"
        '      "name": "<resolved company name>",\n'
        '      "resolved": true | false,\n'
        '      "quarters": [ {"q": "Jun 2022", "sales_cr": 0.0, "net_profit_cr": 0.0}, ... ],\n'
        '      "shareholding": [ {"q": "Jun 2022", "promoters_pct": 0.0, "fii_pct": 0.0, "dii_pct": 0.0, "public_pct": 0.0, "others_pct": 0.0}, ... ]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Companies batch (use name → NSE → BSE symbol → BSE code priority to locate pages):\n"
        + json.dumps(companies, ensure_ascii=False)
    )

def _extract_json(text: str) -> Dict[str, Any]:
    # Try direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict): return obj
    except Exception:
        pass
    # Salvage JSON from text by grabbing the outermost braces
    m1 = text.find("{"); m2 = text.rfind("}")
    if m1 != -1 and m2 != -1 and m2 > m1:
        snip = text[m1:m2+1]
        try:
            obj = json.loads(snip)
            if isinstance(obj, dict): return obj
        except Exception:
            pass
    return {"companies": []}

MAX_LLM_RETRIES = int(os.getenv("MAX_LLM_RETRIES", "3"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "2.0"))
USE_WEB_SEARCH = os.getenv("USE_WEB_SEARCH", "1") not in ("0","false","False")

def call_web_only_json(client: OpenAI, batch_id: int, group: List[Dict[str, str]]) -> Dict[str, Any]:
    prompt = _json_hard_prompt(group)

    last_err = None
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                tools=([{"type": "web_search"}] if USE_WEB_SEARCH else []),
                input=[{"role": "user", "content": prompt}],
            )

            # Save full structured response for inspection
            try:
                (OUTDIR / f"resp_batch_{batch_id:02d}.json").write_text(
                    resp.model_dump_json(indent=2), encoding="utf-8"
                )
            except Exception:
                pass

            # Extract text robustly
            text = _extract_text_from_responses(resp)

            # Save raw text
            try:
                (OUTDIR / f"raw_batch_{batch_id:02d}.txt").write_text(text, encoding="utf-8")
            except Exception:
                pass

            obj = _extract_json(text)
            if isinstance(obj, dict) and isinstance(obj.get("companies"), list):
                return obj
            # If weird content, return empty but stop further retries (this is not transient)
            return {"companies": []}

        except APIStatusError as e:
            # quota/rate-limit/5xx handling
            msg = getattr(e, "message", "") or str(e)
            body = getattr(e, "body", {}) or {}
            code = body.get("code")
            typ  = body.get("type")

            if code == "insufficient_quota" or typ == "insufficient_quota" or "insufficient_quota" in msg:
                # Stop immediately to avoid charges
                raise RuntimeError("INSUFFICIENT_QUOTA") from e

            # Rate limit or 5xx? backoff and retry
            if isinstance(e, RateLimitError) or "rate_limit" in msg or e.status_code in (429, 500, 502, 503, 504):
                last_err = e
            else:
                # non-transient API error
                print("API error (non-transient):", msg)
                return {"companies": []}

        except (APIConnectionError, TimeoutError) as e:
            # transient network/timeout
            last_err = e

        except Exception as e:
            # anything else: log once and return empty
            print("LLM call failed:", e)
            return {"companies": []}

        # Backoff before next retry
        if attempt < MAX_LLM_RETRIES:
            delay = RETRY_BACKOFF_BASE ** attempt
            print(f"Retrying LLM call in {delay:.1f}s (attempt {attempt+1}/{MAX_LLM_RETRIES})…")
            time.sleep(delay)

    # If we get here, all retries failed
    if last_err:
        print("LLM call failed after retries:", last_err)
    return {"companies": []}
# ---------- Tidy / plotting / I/O ----------
def tidy_to_frames(collected: List[Dict[str, Any]]):
    q_rows, s_rows = [], []
    for c in collected:
        name = c.get("name", "")
        if not c.get("resolved", False):  # ignore unresolved
            continue
        for row in (c.get("quarters") or []):
            q_rows.append({"company": name, "quarter": row.get("q",""),
                           "sales_cr": row.get("sales_cr"), "net_profit_cr": row.get("net_profit_cr")})
        for row in (c.get("shareholding") or []):
            s_rows.append({"company": name, "quarter": row.get("q",""),
                           "promoters_pct": row.get("promoters_pct"), "fii_pct": row.get("fii_pct"),
                           "dii_pct": row.get("dii_pct"), "public_pct": row.get("public_pct"),
                           "others_pct": row.get("others_pct")})
    qdf = pd.DataFrame(q_rows, columns=["company","quarter","sales_cr","net_profit_cr"])
    sdf = pd.DataFrame(s_rows, columns=["company","quarter","promoters_pct","fii_pct","dii_pct","public_pct","others_pct"])
    return qdf, sdf

def plot_company(pdf: PdfPages, name: str, qdf: pd.DataFrame, sdf: pd.DataFrame):
    plt.figure(figsize=(10,5))
    sub = qdf[qdf["company"]==name].sort_values("quarter")
    if sub.empty:
        plt.title(f"{name} — Sales & Net Profit (no data)")
    else:
        plt.plot(sub["quarter"], sub["sales_cr"], label="Sales (₹ Cr)")
        plt.plot(sub["quarter"], sub["net_profit_cr"], label="Net Profit (₹ Cr)")
        plt.title(f"{name} — Sales vs Net Profit")
        plt.xticks(rotation=45, ha="right"); plt.xlabel("Quarter"); plt.ylabel("₹ Cr")
        plt.legend(); plt.tight_layout()
    pdf.savefig(); plt.close()

    plt.figure(figsize=(10,5))
    sub = sdf[sdf["company"]==name].sort_values("quarter")
    if sub.empty:
        plt.title(f"{name} — Shareholding % (no data)")
    else:
        for col in ["promoters_pct","fii_pct","dii_pct","public_pct","others_pct"]:
            if col in sub and sub[col].notna().any():
                plt.plot(sub["quarter"], sub[col], label=col.replace("_"," ").title())
        plt.title(f"{name} — Shareholding %")
        plt.xticks(rotation=45, ha="right"); plt.xlabel("Quarter"); plt.ylabel("%")
        plt.legend(); plt.tight_layout()
    pdf.savefig(); plt.close()

def write_batch_pdf(batch_id: int, names: List[str], qdf: pd.DataFrame, sdf: pd.DataFrame) -> pathlib.Path:
    out_pdf = OUTDIR / f"Report_batch_{batch_id:02d}.pdf"
    with PdfPages(out_pdf) as pdf:
        if not names:
            plt.figure(figsize=(8,4)); plt.axis("off")
            plt.text(0.5,0.5,f"Batch {batch_id}: no data.", ha="center", va="center", fontsize=12)
            pdf.savefig(); plt.close()
        else:
            for name in names:
                plot_company(pdf, name, qdf, sdf)
    return out_pdf

def append_batch_csv(qdf: pd.DataFrame, sdf: pd.DataFrame):
    merged = pd.merge(qdf, sdf, on=["company","quarter"], how="outer").sort_values(["company","quarter"])
    mode = "a" if MASTER_CSV.exists() else "w"
    merged.to_csv(MASTER_CSV, index=False, encoding="utf-8", mode=mode, header=(mode=="w"))

def merge_all_pdfs(batch_pdfs: List[pathlib.Path], out_path: pathlib.Path):
    if not HAVE_PYPDF:
        print("pypdf not installed; skipping final PDF merge.")
        return
    merger = PdfMerger()
    for p in batch_pdfs:
        if p.exists() and p.stat().st_size > 0:
            merger.append(str(p))
    if len(merger.pages) == 0:
        print("No pages to merge."); return
    with open(out_path, "wb") as f:
        merger.write(f)
    merger.close()
    print("✓ Merged PDF:", out_path)

# ---------- Main ----------
def main():
    client = OpenAI(timeout=REQUEST_TIMEOUT, max_retries=SDK_MAX_RETRIES)
    companies = load_companies_xlsx()

    batch_pdfs: List[pathlib.Path] = []
    for bid, group in chunk(companies, BATCH_SIZE):
        print(f"-- Batch {bid} ({len(group)} companies) --")
        try:
            data = call_web_only_json(client, bid, group)
        except RuntimeError as e:
            if str(e) == "INSUFFICIENT_QUOTA":
                print("Detected insufficient_quota — stopping immediately.")
                break
            raise

        batch_list = data.get("companies", [])
        print(f"   received {len(batch_list)} company payloads")

        # Tidy (works even if batch_list is empty)
        qdf_b, sdf_b = tidy_to_frames(batch_list)

        # Always write a batch PDF so build/ has content
        # (names will be [], so the placeholder “Batch X: no data.” page is created)
        names_q = qdf_b["company"] if "company" in qdf_b else pd.Series(dtype=str)
        names_s = sdf_b["company"] if "company" in sdf_b else pd.Series(dtype=str)
        names = sorted(set(pd.concat([names_q, names_s], ignore_index=True).dropna()))
        pdf_path = write_batch_pdf(bid, names, qdf_b, sdf_b)
        batch_pdfs.append(pdf_path)
        print(f"✓ Batch {bid} PDF:", pdf_path)

        # Append CSV (safe even when empty; produces headers once if file absent)
        try:
            append_batch_csv(qdf_b, sdf_b)
        except Exception as ex:
            print(f"WARN: CSV append failed for batch {bid}: {ex}")

        # If empty and configured to stop, now we have a PDF in build/; stop here
        if not batch_list and STOP_IF_EMPTY:
            print("Empty batch result. Stopping to avoid further charges.")
            break

        time.sleep(1)  # gentle pacing

    # Try merging individual batch PDFs (optional)
    try:
        merge_all_pdfs(batch_pdfs, MASTER_PDF)
    except Exception as ex:
        print(f"WARN: Failed to merge PDFs: {ex}")

    print("\nArtifacts in build/:")
    for p in sorted(OUTDIR.glob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()
