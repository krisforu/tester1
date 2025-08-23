# scripts/run_batches.py
import os, json, time, pathlib, math
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from openai import OpenAI

MODEL = os.getenv("RESP_MODEL", "gpt-5")   # GPT-5
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
OUTDIR = pathlib.Path("build"); OUTDIR.mkdir(parents=True, exist_ok=True)

def load_companies_xlsx(path="data/companies.xlsx") -> List[Dict[str, str]]:
    # Accepts wide sheets; we look for common headers.
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    def pick(*names):
        low = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in low: return low[n.lower()]
        return None
    c_name = pick("Company","Name","Company Name")
    c_nse  = pick("NSE_Symbol","NSE Symbol","NSE")
    c_bses = pick("BSE_Symbol","BSE Symbol","BSE")
    c_bsec = pick("BSE_Code","BSE Code","Code")
    if not c_name:
        raise ValueError("Excel must contain a company name column (e.g., 'Company').")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "name": str(r[c_name]).strip(),
            "nse_symbol": ("" if not c_nse  or pd.isna(r.get(c_nse))  else str(r[c_nse]).strip()),
            "bse_symbol": ("" if not c_bses or pd.isna(r.get(c_bses)) else str(r[c_bses]).strip()),
            "bse_code":   "" if not c_bsec or pd.isna(r.get(c_bsec)) else str(int(r[c_bsec])) if str(r[c_bsec]).strip().isdigit() or str(r[c_bsec]).replace(".0","").isdigit() else str(r[c_bsec]).strip()
        })
    return rows

def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield (i//size)+1, lst[i:i+size]

def call_web_only_json(client: OpenAI, group: List[Dict[str,str]]) -> Dict[str, Any]:
    prompt = open("prompts/PROMPT.md","r",encoding="utf-8").read()
    payload = {"companies": group, "want_quarters":"all"}

    resp = client.responses.create(
        model=MODEL,
        tools=[{"type":"web_search"}],
        response_format={"type": "json_object"},
        input=[
            {"role":"system","content": prompt},
            {"role":"user","content": json.dumps(payload)}
        ]
    )
    # The aggregated JSON is returned as text
    data = {}
    try:
        data = json.loads(getattr(resp, "output_text", "{}") or "{}")
    except Exception as e:
        print("WARN: JSON parse failed:", e)
    return data or {"companies":[]}

def tidy_to_frames(collected: List[Dict[str,Any]]):
    q_rows, s_rows = [], []
    for c in collected:
        name = c.get("name","")
        if not c.get("resolved", False):  # unresolved company
            continue
        for row in c.get("quarters", []) or []:
            q_rows.append({
                "company": name,
                "quarter": row.get("q",""),
                "sales_cr": row.get("sales_cr", None),
                "net_profit_cr": row.get("net_profit_cr", None)
            })
        for row in c.get("shareholding", []) or []:
            s_rows.append({
                "company": name,
                "quarter": row.get("q",""),
                "promoters_pct": row.get("promoters_pct", None),
                "fii_pct": row.get("fii_pct", None),
                "dii_pct": row.get("dii_pct", None),
                "public_pct": row.get("public_pct", None),
                "others_pct": row.get("others_pct", None)
            })
    qdf = pd.DataFrame(q_rows)
    sdf = pd.DataFrame(s_rows)
    return qdf, sdf

def plot_company(pdf: PdfPages, name: str, qdf: pd.DataFrame, sdf: pd.DataFrame):
    # Page 1: Sales & Net Profit
    plt.figure(figsize=(10,5))
    sub = qdf[qdf["company"]==name].sort_values("quarter")
    if sub.empty:
        plt.title(f"{name} — Sales & Net Profit (no data)")
    else:
        plt.plot(sub["quarter"], sub["sales_cr"], label="Sales (₹ Cr)")
        plt.plot(sub["quarter"], sub["net_profit_cr"], label="Net Profit (₹ Cr)")
        plt.title(f"{name} — Sales vs Net Profit")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Quarter")
        plt.ylabel("₹ Cr")
        plt.legend()
        plt.tight_layout()
    pdf.savefig(); plt.close()

    # Page 2: Shareholding pattern
    plt.figure(figsize=(10,5))
    sub = sdf[sdf["company"]==name].sort_values("quarter")
    if sub.empty:
        plt.title(f"{name} — Shareholding % (no data)")
    else:
        for col in ["promoters_pct","fii_pct","dii_pct","public_pct","others_pct"]:
            if col in sub and sub[col].notna().any():
                plt.plot(sub["quarter"], sub[col], label=col.replace("_"," ").title())
        plt.title(f"{name} — Shareholding %")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Quarter"); plt.ylabel("%")
        plt.legend()
        plt.tight_layout()
    pdf.savefig(); plt.close()

def main():
    client = OpenAI()  # uses OPENAI_API_KEY
    companies = load_companies_xlsx()

    all_companies_payload: List[Dict[str,Any]] = []
    for bid, group in chunk(companies, BATCH_SIZE):
        print(f"-- Batch {bid} ({len(group)} companies) --")
        data = call_web_only_json(client, group)
        batch_list = data.get("companies", [])
        print(f"   received {len(batch_list)} company payloads")
        all_companies_payload.extend(batch_list)
        time.sleep(1)

    # Tidy into DataFrames
    qdf, sdf = tidy_to_frames(all_companies_payload)

    # Write master CSV (one file combining quarterly + shareholding)
    # Join so each row has both if available
    out_csv = OUTDIR / "AllCompanies_Quarterly.csv"
    merged = pd.merge(
        qdf, sdf,
        on=["company","quarter"], how="outer"
    ).sort_values(["company","quarter"])
    merged.to_csv(out_csv, index=False, encoding="utf-8")
    print("✓ CSV:", out_csv)

    # Multi-page PDF: two pages per company
    out_pdf = OUTDIR / "AllCompanies_Report.pdf"
    with PdfPages(out_pdf) as pdf:
        for name in sorted(set(merged["company"].dropna())):
            plot_company(pdf, name, qdf, sdf)
    print("✓ PDF:", out_pdf)

    print("\nArtifacts in build/:")
    for p in sorted(OUTDIR.glob("*")):
        print(" -", p)

if __name__ == "__main__":
    main()
