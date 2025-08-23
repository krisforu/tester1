# scripts/run_batches.py
import os, json, math, time, pathlib
import pandas as pd
from typing import List, Dict
from openai import OpenAI

BATCH_SIZE = 5
MODEL = "o4-mini"          # or another Responses API model you prefer
OUTDIR = pathlib.Path("build")
OUTDIR.mkdir(parents=True, exist_ok=True)

import os
import pandas as pd

def load_companies(path="data/companies.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        # try common Excel names
        for x in ["data/companies.xlsx", "data/companies10.xlsx"]:
            if os.path.exists(x):
                df = pd.read_excel(x)
                break
        else:
            raise FileNotFoundError(
                "Provide data/companies.csv or data/companies.xlsx (with columns: name, screener_slug, bse_code)"
            )

    rows = []
    for _, r in df.iterrows():
        slug = ("" if pd.isna(r.get("screener_slug")) else str(r.get("screener_slug")).strip())
        code = ("" if pd.isna(r.get("bse_code")) else str(r.get("bse_code")).strip())
        rows.append({"name": r["name"], "slug_or_code": slug or code})
    return rows

def chunker(seq, size):
    for i in range(0, len(seq), size):
        yield i//size + 1, seq[i:i+size]

def call_llm_for_batch(client, batch_id: int, companies: List[Dict]):
    # Prepare the user content as structured JSON the model can loop over
    user_payload = {
        "batch_id": batch_id,
        "companies": companies
    }

    resp = client.responses.create(
        model=MODEL,
        # Enable tools: web search + code interpreter (official capability)
        tools=[{"type":"web_search"}, {"type":"code_interpreter"}],  # :contentReference[oaicite:4]{index=4}
        input=[
            {"role":"system","content": open("prompts/PROMPT.md","r",encoding="utf-8").read()},
            {"role":"user","content": json.dumps(user_payload)}
        ]
    )

    # Find any files produced by Code Interpreter (PDF/CSV) and download them
    files = []
    for item in resp.output or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") == "output_file" and c.file_id:
                files.append(c.file_id)

    saved = []
    for fid in files:
        fmeta = client.files.retrieve(fid)
        fname = fmeta.filename or f"batch_{batch_id}_{fid}"
        out_path = OUTDIR / fname
        with client.files.with_streaming_response.download(fid) as stream:
            stream.stream_to_file(out_path)
        saved.append(out_path)
    return saved

def main():
    client = OpenAI()  # requires OPENAI_API_KEY in env
    companies = load_companies()
    all_pdfs = []
    for batch_id, batch in chunker(companies, BATCH_SIZE):
        saved = call_llm_for_batch(client, batch_id, batch)
        print(f"[batch {batch_id}] saved:", [str(p) for p in saved])
        # collect PDFs for final merge
        for p in saved:
            if p.suffix.lower() == ".pdf":
                all_pdfs.append(str(p))

    print("\nArtifacts in build/:")
    for p in sorted(OUTDIR.glob("*")):
        print(" -", p)

if __name__ == "__main__":
    main()
