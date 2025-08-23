#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, time, math, pathlib
from typing import List, Dict, Any, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")                    # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from openai import OpenAI
from openai import APIStatusError, APIConnectionError, RateLimitError

# ──────────────────────────────────────────────────────────────────────────────
# Env knobs (safe defaults; override from GitHub Actions env)
# ──────────────────────────────────────────────────────────────────────────────
MODEL              = os.getenv("RESP_MODEL", "gpt-5")
BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "20"))
REQUEST_TIMEOUT    = int(os.getenv("REQUEST_TIMEOUT", "120"))
SDK_MAX_RETRIES    = int(os.getenv("SDK_MAX_RETRIES", "3"))
MAX_LLM_RETRIES    = int(os.getenv("MAX_LLM_RETRIES", "3"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "2.0"))

STOP_IF_EMPTY      = os.getenv("STOP_IF_EMPTY", "1") not in ("0","false","False")
USE_WEB_SEARCH     = os.getenv("USE_WEB_SEARCH", "1") not in ("0","false","False")

OUTDIR = pathlib.Path("build")
OUTDIR.mkdir(parents=True, exist_ok=True)

MASTER_PDF  = OUTDIR / "AllCompanies_Report.pdf"
MASTER_CSV  = OUTDIR / "AllCompanies_Quarterly.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def chunk(seq, size):
    for i in range(0, len(seq), size):
        yield i // size + 1, seq[i : i + size]

def _coerce_bse_code(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def load_companies_xlsx() -> List[Dict[str, str]]:
    """
    Accepts data/companies.xlsx (or companies10.xlsx) or data/companies.csv.
    Expected headers (flexible naming): Company/Name, NSE_Symbol, BSE_Symbol, BSE_Code
    """
    path = None
    for p in ("data/companies.xlsx", "data/companies10.xlsx", "data/companies.csv"):
        if os.path.exists(p):
            path = p
            break
    if not path:
        raise FileNotFoundError("No companies file found (data/companies.xlsx or companies10.xlsx or companies.csv).")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df.columns = [str(c).strip() for c in df.columns]

    def pick(*names):
        low = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in low:
                return low[n.lower()]
        return None

    c_name = pick("Company","Name","Company Name")
    c_nse  = pick("NSE_Symbol","NSE Symbol","NSE")
    c_bss  = pick("BSE_Symbol","BSE Symbol","BSE")
    c_bsc  = pick("BSE_Code","BSE Code","Code")

    if not c_name:
        raise ValueError("Excel must have a company name column (e.g., 'Company').")

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "name":       str(r[c_name]).strip(),
            "nse_symbol": "" if not c_nse or pd.isna(r.get(c_nse))  else str(r[c_nse]).strip(),
            "bse_symbol": "" if not c_bss or pd.isna(r.get(c_bss)) else str(r[c_bss]).strip(),
            "bse_code":   "" if not c_bsc or pd.isna(r.get(c_bsc)) else _coerce_bse_code(r[c_bsc]),
        })
    return rows

def _json_hard_prompt(group: List[Dict[str, str]]) -> str:
    """
    System/user content that instructs the model to produce ONLY JSON,
    and how to resolve Screener pages by name→NSE→BSE symbol→BSE code.
    """
    return (
        "You are a data extractor for screener.in. For each input company, locate the "
        "correct Screener page using this fallback order: (1) Company Name, (2) NSE "
        "symbol, (3) BSE symbol, (4) BSE code. If not found, mark resolved=false.\n\n"
        "For each found company, extract:\n"
        "  • Quarterly Results table → {q, sales_cr, net_profit_cr} for all quarters available\n"
        "  • Shareholding Pattern → {q, promoters_pct, fii_pct, dii_pct, public_pct, others_pct}\n\n"
        "Return ONLY a single JSON object of the following shape (no prose, no explanation):\n"
        "{\n"
        '  "companies": [\n'
        "    {\n"
        '      "name": "<string>",\n'
        '      "resolved": <true|false>,\n'
        '      "quarters": [ {"q":"Jun 2025","sales_cr":123.45,"net_profit_cr":67.89}, ... ],\n'
        '      "shareholding": [ {"q":"Jun 2025","promoters_pct":55.0,"fii_pct":10.0,"dii_pct":5.0,"public_pct":30.0,"others_pct":0.0}, ... ]\n'
        "    }, ...\n"
        "  ]\n"
        "}\n\n"
        f"INPUT_COMPANIES_JSON={json.dumps(group, ensure_ascii=False)}\n"
        "IMPORTANT: respond with JSON ONLY."
    )

def _extract_text_from_responses(resp) -> str:
    """
    Collect text from OpenAI Responses API objects across SDK versions.
    """
    # 1) convenience
    text = getattr(resp, "output_text", None)
    if text:
        return text

    # 2) walk output->content->text
    try:
        chunks = []
        output = getattr(resp, "output", None) or []
        for item in output:
            content = getattr(item, "content", None) or []
            for c in content:
                t = getattr(c, "text", None)
                if t:
                    chunks.append(t)
                elif isinstance(c, dict) and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
        if chunks:
            return "\n".join(chunks)
    except Exception:
        pass

    # 3) brute: dump and find all "text"
    try:
        dumped = resp.model_dump()
        def find_texts(n):
            res = []
            if isinstance(n, dict):
                for k,v in n.items():
                    if k == "text" and isinstance(v, str):
                        res.append(v)
                    else:
                        res.extend(find_texts(v))
            elif isinstance(n, list):
                for v in n:
                    res.extend(find_texts(v))
            return res
        texts = find_texts(dumped)
        if texts:
            return "\n".join(texts)
    except Exception:
        pass

    return ""

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Salvage the first {...} object from a string and parse it.
    """
    if not text:
        return {}
    # If it's already JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first JSON object via brace matching
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return {}

def call_web_only_json(client: OpenAI, batch_id: int, group: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Call the Responses API with retries + backoff.
    Saves both the structured response and the raw text for debugging.
    """
    prompt = _json_hard_prompt(group)
    last_err = None

    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                tools=([{"type": "web_search"}] if USE_WEB_SEARCH else []),
                input=[{"role": "user", "content": prompt}],
            )

            # Save full structured response
            try:
                (OUTDIR / f"resp_batch_{batch_id:02d}.json").write_text(
                    resp.model_dump_json(indent=2), encoding="utf-8"
                )
            except Exception:
                pass

            text = _extract_text_from_responses(resp)

            # Save raw text
            try:
                (OUTDIR / f"raw_batch_{batch_id:02d}.txt").write_text(text, encoding="utf-8")
            except Exception:
                pass

            obj = _extract_json(text)
            if isinstance(obj, dict) and isinstance(obj.get("companies"), list):
                return obj
            return {"companies": []}  # successful call, but unusable content → don't retry

        except APIStatusError as e:
            msg  = getattr(e, "message", "") or str(e)
            body = getattr(e, "body", {}) or {}
            code = body.get("code"); typ = body.get("type")

            if code == "insufficient_quota" or typ == "insufficient_quota" or "insufficient_quota" in msg:
                raise RuntimeError("INSUFFICIENT_QUOTA") from e

            # 429 or 5xx? backoff
            if getattr(e, "status_code", None) in (429, 500, 502, 503, 504) or "rate_limit" in msg:
                last_err = e
            else:
                print("API error (non-transient):", msg)
                return {"companies": []}

        except (APIConnectionError, TimeoutError) as e:
            last_err = e

        except Exception as e:
            print("LLM call failed:", e)
            return {"companies": []}

        if attempt < MAX_LLM_RETRIES:
            delay = RETRY_BACKOFF_BASE ** attempt
            print(f"Retrying LLM call in {delay:.1f}s (attempt {attempt+1}/{MAX_LLM_RETRIES})…")
            time.sleep(delay)

    if last_err:
        print("LLM call failed after retries:", last_err)
    return {"companies": []}

def tidy_to_frames(collected: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    q_rows, s_rows = [], []
    for c in collected or []:
        name = c.get("name", "")
        if not c.get("resolved", False):
            # keep unresolved out of charts
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

def _plot_company_page(pdf: PdfPages, name: str, qdf: pd.DataFrame, sdf: pd.DataFrame):
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
        plt.xlabel("Quarter")
        plt.ylabel("%")
        plt.legend()
        plt.tight_layout()
    pdf.savefig(); plt.close()

def write_batch_pdf(batch_id: int, names: List[str], qdf: pd.DataFrame, sdf: pd.DataFrame) -> pathlib.Path:
    out = OUTDIR / f"Report_batch_{batch_id:02d}.pdf"
    with PdfPages(out) as pdf:
        if not names:
            # placeholder page so artifact upload never fails
            plt.figure(figsize=(10,5))
            plt.text(0.5, 0.6, f"Batch {batch_id}: no data.", ha="center", va="center", fontsize=32)
            plt.axis("off")
            pdf.savefig(); plt.close()
        else:
            for n in names:
                _plot_company_page(pdf, n, qdf, sdf)
    return out

def append_batch_csv(qdf: pd.DataFrame, sdf: pd.DataFrame):
    merged = pd.merge(qdf, sdf, on=["company","quarter"], how="outer")
    header = not MASTER_CSV.exists()
    with open(MASTER_CSV, "a", encoding="utf-8") as fh:
        merged.to_csv(fh, index=False, header=header)

def merge_all_pdfs(batch_paths: List[pathlib.Path], out_path: pathlib.Path):
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception:
        print("pypdf not installed; skipping final PDF merge.")
        return
    writer = PdfWriter()
    for p in batch_paths:
        try:
            r = PdfReader(str(p))
            for page in r.pages:
                writer.add_page(page)
        except Exception as e:
            print(f"WARN: failed to read {p}: {e}")
    if writer.pages:
        with open(out_path, "wb") as f:
            writer.write(f)
        print("✓ Merged PDF:", out_path)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

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

        # tidy (safe when empty)
        qdf_b, sdf_b = tidy_to_frames(batch_list)

        # determine the names for this batch
        names_q = qdf_b["company"] if "company" in qdf_b else pd.Series(dtype=str)
        names_s = sdf_b["company"] if "company" in sdf_b else pd.Series(dtype=str)
        names = sorted(set(pd.concat([names_q, names_s], ignore_index=True).dropna()))

        # write the batch PDF FIRST so artifacts are guaranteed
        pdf_path = write_batch_pdf(bid, names, qdf_b, sdf_b)
        batch_pdfs.append(pdf_path)
        print(f"✓ Batch {bid} PDF:", pdf_path)

        # append CSV (headers on first write)
        try:
            append_batch_csv(qdf_b, sdf_b)
        except Exception as ex:
            print(f"WARN: CSV append failed for batch {bid}: {ex}")

        # cost-saver: if the batch returned nothing, stop now (we already wrote a PDF)
        if not batch_list and STOP_IF_EMPTY:
            print("Empty batch result. Stopping to avoid further charges.")
            break

        time.sleep(1)  # gentle pacing

    # merge all PDFs
    merge_all_pdfs(batch_pdfs, MASTER_PDF)

    print("\nArtifacts in build/:")
    for p in sorted(OUTDIR.glob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()
