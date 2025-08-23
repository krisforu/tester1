# scripts/run_batches.py
import os
import json
import time
import pathlib
from typing import List, Dict, Optional

import pandas as pd
from openai import OpenAI

# ====== Configuration ======
MODEL = os.getenv("RESP_MODEL", "gpt-5")   # use GPT-5
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "25"))
OUTDIR = pathlib.Path("build")
OUTDIR.mkdir(parents=True, exist_ok=True)


def _col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def load_companies(path_xlsx: str = "data/companies.xlsx") -> List[Dict]:
    """
    Read your wide Excel with many columns and return dicts:
      {name, nse_symbol, bse_symbol, bse_code}
    We do NOT decide the URL here; the prompt will try Company → NSE → BSE symbol → BSE code.
    """
    xpaths = [path_xlsx, "data/companies10.xlsx", "data/companies.xlsm"]
    xfile = next((p for p in xpaths if os.path.exists(p)), None)
    if not xfile:
        raise FileNotFoundError("Expected data/companies.xlsx (or companies10.xlsx)")

    df = pd.read_excel(xfile)  # requires openpyxl
    df.columns = [str(c).strip() for c in df.columns]

    name_c = _col(df, ["Company", "Name", "Company Name"])
    nse_c  = _col(df, ["NSE_Symbol", "NSE", "NSE Ticker", "NSE Symbol"])
    bse_s  = _col(df, ["BSE_Symbol", "BSE", "BSE Ticker", "BSE Symbol"])
    bse_c  = _col(df, ["BSE_Code", "BSE Code", "Code", "BSEcode", "BSEID"])

    if not name_c:
        raise ValueError("Excel must contain a company name column (e.g., 'Company').")

    def clean_bse_code(v):
        if pd.isna(v):
            return ""
        try:
            return str(int(v))  # avoid '543693.0'
        except Exception:
            return str(v).strip()

    rows: List[Dict] = []
    for _, r in df.iterrows():
        rows.append({
            "name": str(r[name_c]).strip(),
            "nse_symbol": ("" if not nse_c or pd.isna(r.get(nse_c)) else str(r[nse_c]).strip()),
            "bse_symbol": ("" if not bse_s or pd.isna(r.get(bse_s)) else str(r[bse_s]).strip()),
            "bse_code": clean_bse_code(r[bse_c]) if bse_c else ""
        })
    return rows


def chunk(lst: List[Dict], size: int):
    for i in range(0, len(lst), size):
        yield (i // size) + 1, lst[i:i + size]


def call_batch(client: OpenAI, batch_id: int, companies: List[Dict]) -> List[pathlib.Path]:
    payload = {
        "companies": companies,
        "out_pdf": f"Report_batch_{batch_id:02d}.pdf",
        "out_csv": f"Quarterly_batch_{batch_id:02d}.csv",
    }

    resp = client.responses.create(
        model=MODEL,
        tools=[
            {"type": "web_search"},
            {"type": "code_interpreter", "container": {"type": "auto"}}
        ],
        input=[
            {
                "role": "system",
                "content": open("prompts/PROMPT.md", "r", encoding="utf-8").read()
            },
            {
                "role": "user",
                "content": json.dumps(payload)
            }
        ],
    )

    # Debug (helps when no files are returned)
    print("DEBUG text:\n", getattr(resp, "output_text", ""))
    saved: List[pathlib.Path] = []

    for item in (resp.output or []):
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") == "output_file" and getattr(c, "file_id", None):
                fmeta = client.files.retrieve(c.file_id)
                out_path = OUTDIR / (fmeta.filename or c.file_id)
                with client.files.with_streaming_response.download(c.file_id) as s:
                    s.stream_to_file(out_path)
                print("saved:", out_path)
                saved.append(out_path)
    return saved


def merge_csvs_to_one(out_csv="build/AllCompanies_Quarterly.csv"):
    dfs = []
    for p in sorted(OUTDIR.glob("Quarterly_batch_*.csv")):
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print("WARN: skip CSV", p, e)
    if dfs:
        big = pd.concat(dfs, ignore_index=True)
        big.to_csv(out_csv, index=False, encoding="utf-8")
        print("✓ merged CSV ->", out_csv)


def merge_pdfs_to_one(out_pdf="build/AllCompanies_Report.pdf"):
    try:
        from PyPDF2 import PdfMerger
        merger = PdfMerger()
        any_input = False
        for p in sorted(OUTDIR.glob("Report_batch_*.pdf")):
            merger.append(str(p))
            any_input = True
        if any_input:
            with open(out_pdf, "wb") as f:
                merger.write(f)
            merger.close()
            print("✓ merged PDF ->", out_pdf)
    except Exception as e:
        print("WARN: PDF merge skipped:", e)


def main():
    client = OpenAI()  # uses OPENAI_API_KEY env
    companies = load_companies()

    # Run in batches to avoid container timeouts; tweak BATCH_SIZE via env
    for bid, group in chunk(companies, BATCH_SIZE):
        print(f"== Batch {bid} ({len(group)} companies) ==")
        call_batch(client, bid, group)
        time.sleep(2)  # polite pacing

    # Merge all batches into one CSV + one PDF
    merge_csvs_to_one()
    merge_pdfs_to_one()

    print("\nArtifacts in build/:")
    for p in sorted(OUTDIR.glob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()
