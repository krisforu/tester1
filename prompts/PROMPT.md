You are an agent with Web Search and Code Interpreter enabled.

Goal: For each company in the {batch}, visit its Screener page:
  - URL: https://www.screener.in/company/{slug_or_code}/
  - From “Quarterly Results”: extract Sales and Net Profit for 13 quarters
    (Jun 2022 → Jun 2025).
  - From “Investors → Shareholding Pattern”: extract Promoter %, DII %, Public % for
    the same set of quarters (if DII missing, treat as 0).
  - Produce TWO matplotlib line charts per company:
      (1) Sales vs Net Profit
      (2) Shareholding Pattern (Promoter/DII/Public)
  - Save charts to a multi-page PDF named batch_{batch_id}.pdf (two pages per company),
    and also return a CSV of tidy rows:
       Company, Quarter, Sales, NetProfit, Promoter, DII, Public.

Implementation rules:
  - Use Web Search to load each Screener page.
  - Use Python in Code Interpreter to parse HTML tables (pandas.read_html),
    align/clean the last 13 quarters, draw charts, and write the PDF/CSV.
  - Be polite to the site (don’t hammer it) and handle missing values.
Return: the generated PDF file and CSV file as outputs.

Output rules (IMPORTANT):

- Use Python in Code Interpreter to generate TWO files and save them with EXACT names:
  • AllCompanies_Report.pdf
  • AllCompanies_Quarterly.csv

- After saving, ATTACH both files to your final message so they appear as downloadable files
  (do not paste CSV inline; return them as files).

- If a table is missing for a company, still create the files and mark that company as "missing".
