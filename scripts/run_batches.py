# scripts/run_batches.py
import json
import pathlib
from typing import List, Dict

import pandas as pd
from openai import OpenAI

MODEL = "o4-mini"  # Responses API model
OUTDIR = pathlib.Path("build")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_companies(path: str = "data/companies.csv") -> List[Dict]:
    df = pd.read_csv(path)
    rows: List[Dict] = []
    for _, r in df.iterrows():
        slug = ("" if pd.isna(r.get("screener_slug")) else str(r.get("screener_slug")).strip())
        code = ("" if pd.isna(r.get("bse_code")) else str(r.get("bse_code")).strip())
        rows.append({"name": r["name"], "slug_or_code": slug or code})
    return rows


def call_llm_for_batch(client: OpenAI, batch_id: int, companies: List[Dict]):
    # We still call it “batch” for compatibility, but we pass ALL companies
    user_payload = {
        "batch_id": batch_id,
        "companies": companies,
        "out_pdf": "AllCompanies_Report.pdf",
        "out_csv": "AllCompanies_Quarterly.csv",
    }

    resp = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search"}, {"type": "code_interpreter"}],
        input=[
            {"role": "system", "content": open("prompts/PROMPT.md", "r", encoding="utf-8").read()},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    )

    # Download files produced by Code Interpreter
    saved = []
    for item in (resp.output or []):
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") == "output_file" and getattr(c, "file_id", None):
                fmeta = client.files.retrieve(c.file_id)
                fname = fmeta.filename or f"batch_{batch_id}_{c.file_id}"
                out_path = OUTDIR / fname
                with client.files.with_streaming_response.download(c.file_id) as stream:
                    stream.stream_to_file(out_path)
                saved.append(out_path)
    return saved


def main():
    client = OpenAI()  # needs OPENAI_API_KEY
    companies = load_companies()
    saved = call_llm_for_batch(client, 1, companies)
    print("[all] saved:", [str(p) for p in saved])

    print("\nArtifacts in build/:")
    for p in sorted(OUTDIR.glob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()
