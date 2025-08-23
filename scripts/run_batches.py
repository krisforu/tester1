# scripts/run_batches.py
import os, json, time, pathlib
from typing import List, Dict, Any

import pandas as pd

# Headless plotting for CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from openai import OpenAI
from openai import APIConnectionError  # for precise retry targeting

# ====== Config ======
MODEL = os.getenv("RESP_MODEL", "gpt-5")           # set in workflow env
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))    # keep modest to reduce rework on failure
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "90.0"))  # seconds
SDK_MAX_RETRIES = int(os.getenv("SDK_MAX_RETRIES", "2"))       # SDK-level retries (short)

OUTDIR = pathlib.Path("build")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers to read your wide Excel ----------
def _pick(df: pd.DataFrame, *candidates: str) -> str | None:
    low = {str(c).strip().lower(): c for c in df.columns}
    for n in candidates:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _find_excel() -> str:
    for p in ("data/companies.xlsx", "data/companies10.xlsx"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Expected data/companies.xlsx (or data/companies10.xlsx).")


def _clean_bse_code(v) -> str:
    """Return '543693' from values such as 543693, '543693', or '543693.0'."""
    if pd.isna(v):
        return ""
    s = str(v).strip()
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_companies_xlsx() -> List[Dict[str, str]]:
    """
    Reads a wide Excel and returns rows with:
      {name, nse_symbol, bse_symbol, bse_code}
    The prompt will try Company → NSE → BSE Symbol → BSE Code.
    """
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
            "nse_symbol": "" if not c_nse  or pd.isna(r.get(c_nse))  else str(r[c_nse]).strip(),
            "bse_symbol": "" if not c_bses or pd.isna(r.get(c_bses)) else str(r[c_bses]).strip(),
            "bse_code":   "" if not c_bsec or pd.isna(r.get(c_bsec)) else _clean_bse_code(r[c_bsec]),
        })
    return rows


def chunk(lst: List[Dict[str, str]], size: int):
    for i in range(0, len(lst), size):
        yield (i // size) + 1, lst[i:i + size]


# ---------- LLM call with robust retry ----------
# Cookbook recommends exponential backoff for transient errors. :contentReference[oaicite:1]{index=1}
@retry(
    reraise=True,
    stop=stop_after_attempt(5),                              # up to 5 attempts
    wait=wait_random_exponential(multiplier=1, max=20),     # 1s → ~20s backoff with jitter
    retry=retry_if_exception_type((
        APIConnectionError,          # OpenAI SDK network error
        httpx.ReadTimeout,           # transport timeout
        httpx.RemoteProtocolError,   # "Server disconnected without sending a response"
        httpx.ConnectError,
        httpx.NetworkError,
    ))
)
def _responses_create_with_retry(client: OpenAI, *, model: str, tools: list, input: list):
    return client.responses.create(
        model=model,
        tools=tools,
        input=input,
    )


def call_web_only_json(client: OpenAI, group: List[Dict[str, str]]) -> Dict[str, Any]:
    prompt = open("prompts/PROMPT.md", "r", encoding="utf-8").read()
    payload = {"companies": group, "want_quarters": "all"}

    # One call per batch, retried on transient network failures
    resp = _responses_create_with_retry(
        client,
        model=MODEL,
        tools=[{"type": "web_search"}],
        input=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": json.dumps(payload)}
        ],
    )

    text = getattr(resp, "output_text", "") or "{}"
    try:
        return json.loads(text)
    except Exception as e:
        print("WARN: JSON parse failed:", e)
        print("RAW TEXT (first 2k):", text[:2000])
        return {"companies": []}


# ---------- Tidy into DataFrames ----------
def tidy_to_frames(collected: List[Dict[str, Any]]):
    q_rows, s_rows = [], []
    for c in collected:
        name = c.get("name", "")
        if not c.get("resolved", False):
            continue
        for row in (c.get("quarters") or []):
            q_rows.append({
                "company": name,
                "quarter": row.get("q", ""),
                "sales_cr": row.get("sales_cr", None),
                "net_profit_cr": row.get("net_profit_cr", None),
            })
        for row in (c.get("shareholding") or []):
            s_rows.append({
                "company": name,
                "quarter": row.get("q", ""),
                "promoters_pct": row.get("promoters_pct", None),
                "fii_pct": row.get("fii_pct", None),
                "dii_pct": row.get("dii_pct", None),
                "public_pct": row.get("public_pct", None),
                "others_pct": row.get("others_pct", None),
            })
    return pd.DataFrame(q_rows), pd.DataFrame(s_rows)


# ---------- Plotting ----------
def plot_company(pdf: PdfPages, name: str, qdf: pd.DataFrame, sdf: pd.DataFrame):
    # Page 1: Sales & Net Profit
    plt.figure(figsize=(10, 5))
    sub = qdf[qdf["company"] == name].sort_values("quarter")
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

    # Page 2: Shareholding %
    plt.figure(figsize=(10, 5))
    sub = sdf[sdf["company"] == name].sort_values("quarter")
    if sub.empty:
        plt.title(f"{name} — Shareholding % (no data)")
    else:
        for col in ["promoters_pct", "fii_pct", "dii_pct", "public_pct", "others_pct"]:
            if col in sub and sub[col].notna().any():
                plt.plot(sub["quarter"], sub[col], label=col.replace("_", " ").title())
        plt.title(f"{name} — Shareholding %")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Quarter"); plt.ylabel("%")
        plt.legend()
        plt.tight_layout()
    pdf.savefig(); plt.close()


# ---------- main ----------
def main():
    # Increase SDK-level timeout & short internal retries to reduce spurious failures. :contentReference[oaicite:2]{index=2}
    client = OpenAI(timeout=REQUEST_TIMEOUT, max_retries=SDK_MAX_RETRIES)

    companies = load_companies_xlsx()

    all_companies_payload: List[Dict[str, Any]] = []
    for bid, group in chunk(companies, BATCH_SIZE):
        print(f"-- Batch {bid} ({len(group)} companies) --")
        try:
            data = call_web_only_json(client, group)
        except Exception as e:
            # After all retry attempts, fail *softly* for this batch so you don't burn more tokens.
            print(f"ERROR: batch {bid} failed after retries: {e}")
            data = {"companies": []}
        batch_list = data.get("companies", [])
        print(f"   received {len(batch_list)} company payloads")
        all_companies_payload.extend(batch_list)
        time.sleep(1)  # gentle pacing

    # Tidy → DataFrames
    qdf, sdf = tidy_to_frames(all_companies_payload)

    # Combined CSV (quarterly + shareholding)
    out_csv = OUTDIR / "AllCompanies_Quarterly.csv"
    merged = pd.merge(qdf, sdf, on=["company", "quarter"], how="outer").sort_values(["company", "quarter"])
    merged.to_csv(out_csv, index=False, encoding="utf-8")
    print("✓ CSV:", out_csv)

    # Multi-page PDF
    out_pdf = OUTDIR / "AllCompanies_Report.pdf"
    with PdfPages(out_pdf) as pdf:
        names_q = qdf["company"] if "company" in qdf else pd.Series(dtype=str)
        names_s = sdf["company"] if "company" in sdf else pd.Series(dtype=str)
        companies_for_pdf = sorted(set(pd.concat([names_q, names_s], ignore_index=True).dropna()))

        if not companies_for_pdf:
            plt.figure(figsize=(8, 4))
            plt.axis("off")
            plt.text(0.5, 0.5, "No data returned (network or source issue).", ha="center", va="center", fontsize=12)
            pdf.savefig(); plt.close()
        else:
            for name in companies_for_pdf:
                plot_company(pdf, name, qdf, sdf)
    print("✓ PDF:", out_pdf)

    print("\nArtifacts in build/:")
    for p in sorted(OUTDIR.glob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()
