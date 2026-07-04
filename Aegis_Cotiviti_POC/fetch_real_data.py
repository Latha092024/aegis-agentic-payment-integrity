"""
fetch_real_data.py  -  RUN THIS ON YOUR MACHINE (not needed inside the demo).

Pulls REAL charge rows from the CMS "Medicare Physician & Other Practitioners
by Provider and Service" public API and writes data/real/cms_charges_full.csv.
Aegis then automatically prefers this fuller real data over the shipped sample.

Source (public domain, US government works):
  https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service

Usage:
  pip install requests
  python fetch_real_data.py
"""
import csv
import os
import time

import requests

# 2023 (latest) distribution id from the dataset's data.gov metadata.
# If CMS publishes a newer year, grab the new id from the dataset's "API" tab.
DATASET_ID = "92396110-2aed-4d63-a6a2-5d6207d46a29"
BASE = f"https://data.cms.gov/data-api/v1/dataset/{DATASET_ID}/data"

# The codes used by the demo claims (extend freely).
CODES = ["99213", "99214", "99215", "80053", "80048", "20610",
         "93307", "11720", "11055", "36415", "85025"]

KEEP = ["Rndrng_NPI", "Rndrng_Prvdr_Type", "HCPCS_Cd", "Place_Of_Srvc",
        "Tot_Srvcs", "Avg_Sbmtd_Chrg"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "real")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "cms_charges_full.csv")


def fetch_code(code, page=5000, cap=20000):
    """Page through the API for one HCPCS code (server-side filter)."""
    rows, offset = [], 0
    while offset < cap:
        params = {"filter[HCPCS_Cd]": code, "size": page, "offset": offset}
        r = requests.get(BASE, params=params, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
        time.sleep(0.3)
    return rows


def main():
    all_rows = []
    for code in CODES:
        try:
            got = fetch_code(code)
            all_rows.extend(got)
            print(f"  {code}: {len(got)} rows")
        except Exception as e:
            print(f"  {code}: FAILED ({e})")
    if not all_rows:
        print("No data fetched. Check your connection or the DATASET_ID.")
        return
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEEP, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, "") for k in KEEP})
    print(f"Wrote {len(all_rows)} real rows -> {OUT}")
    print("Aegis will now use these real baselines automatically.")


if __name__ == "__main__":
    main()
