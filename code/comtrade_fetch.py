"""
Pull the real demand panel for the Turkey cement/clinker export case study from
the UN Comtrade *public preview* API (no subscription key required).

Run this LOCALLY (it needs internet; the response for a full year of all
partners is large, so we query year-by-year and aggregate). It writes
``case_data/demand_panel_long.csv`` with columns
    year, product_hs, partner_code, partner_name, tons, value_usd
which ``case_loader.py`` turns into the model's demand samples.

Why preview: the endpoint
  https://comtradeapi.un.org/public/v1/preview/C/A/HS?reporterCode=792&period=YYYY&cmdCode=HS&flowCode=X
returns real partner-level data without an API key. Each partner appears in
several breakdown rows (by customs regime / mode of transport); the fully
aggregated total is the row with customsCode='C00', motCode=0, partner2Code=0
and partnerCode != 0 (0 = World).
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

REPORTER_TURKEY = 792
# Optional free Comtrade subscription key (register at comtradedeveloper.un.org).
# If set, the full data endpoint is used (no preview row cap). Leave "" to use the
# key-free public preview (works, but caps ~ a few hundred rows per call).
API_KEY = ""
# cement family (HS 6-digit); edit to taste / add minerals for a broader portfolio
HS_PRODUCTS = {
    "252310": "clinker",
    "252329": "portland_grey",
    "252321": "portland_white",
    "252010": "gypsum",          # crude gypsum/anhydrite (bulk mineral)
    # "252390": "other_hydraulic",  # dropped: no Turkish data after ~2013
}
YEARS = list(range(2005, 2024))           # ~19 annual observations
OUT = Path(__file__).parent / "case_data"

# minimal M49 numeric -> name map (extend as needed; unmapped codes kept as code)
M49 = {
    842: "USA", 376: "Israel", 288: "Ghana", 384: "CotedIvoire", 120: "Cameroon",
    686: "Senegal", 324: "Guinea", 434: "Libya", 760: "Syria", 642: "Romania",
    724: "Spain", 818: "Egypt", 332: "Haiti", 214: "DominicanRep", 604: "Peru",
    170: "Colombia", 270: "Gambia", 624: "GuineaBissau", 478: "Mauritania",
    768: "Togo", 204: "Benin", 50: "Bangladesh", 144: "SriLanka", 826: "UK",
    392: "Japan", 504: "Morocco", 512: "Oman", 8: "Albania", 12: "Algeria",
}


def fetch_year_product(year: int, hs: str, pause: float = 6.0,
                       retries: int = 6) -> list[dict]:
    base = ("https://comtradeapi.un.org/data/v1/get/C/A/HS" if API_KEY
            else "https://comtradeapi.un.org/public/v1/preview/C/A/HS")
    # motCode=0 asks for the all-modes TOTAL only -> one row per partner, which
    # keeps the response small and avoids the preview row cap.
    url = (f"{base}?reporterCode={REPORTER_TURKEY}&period={year}"
           f"&cmdCode={hs}&flowCode=X&motCode=0")
    headers = {"User-Agent": "research/1.0"}
    if API_KEY:
        headers["Ocp-Apim-Subscription-Key"] = API_KEY
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                payload = json.loads(r.read().decode())
            time.sleep(pause)                      # be polite to the API
            return payload.get("data", [])
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = 30 * (attempt + 1)          # backoff: 30, 60, 90, ... s
                print(f"    429 rate-limited on {year} {hs}; waiting {wait}s...")
                time.sleep(wait)
                continue
            raise


def aggregate_partner_totals(records: list[dict]) -> list[dict]:
    """Partner-level totals. The preview API reports, for each partner, the rows
    by mode of transport (motCode 1000/2100/...) plus a ``motCode == 0`` row that
    is the all-modes TOTAL under customs total C00 -- that row is exactly the
    partner total we want (validated: it equals the sum of the mode rows)."""
    from collections import defaultdict
    by_partner = defaultdict(list)
    for x in records:
        pc = x.get("partnerCode")
        if pc in (0, None) or x.get("customsCode") != "C00" or not x.get("netWgt"):
            continue
        by_partner[pc].append(x)

    out = []
    for pc, xs in by_partner.items():
        totals = [x for x in xs if x.get("motCode") == 0]
        if totals:                                  # use the all-modes total row
            x = max(totals, key=lambda r: r["netWgt"])
            tons, val = x["netWgt"] / 1000.0, x.get("primaryValue") or 0.0
        else:                                       # fall back: sum the mode rows
            seen, tons, val = set(), 0.0, 0.0
            for x in xs:
                key = (x.get("motCode"), x.get("mosCode"), x.get("netWgt"))
                if key in seen:
                    continue
                seen.add(key)
                tons += x["netWgt"] / 1000.0
                val += x.get("primaryValue") or 0.0
        out.append({"partner_code": pc, "partner_name": M49.get(pc, str(pc)),
                    "tons": round(tons, 3), "value_usd": round(val, 2)})
    return out


def build_panel():
    OUT.mkdir(exist_ok=True)
    rows = []
    for hs, name in HS_PRODUCTS.items():
        for yr in YEARS:
            try:
                recs = aggregate_partner_totals(fetch_year_product(yr, hs))
            except Exception as exc:
                print(f"  {yr} {hs} failed: {exc}")
                continue
            for r in recs:
                rows.append([yr, hs, name, r["partner_code"], r["partner_name"],
                             round(r["tons"], 3), round(r["value_usd"], 2)])
            print(f"  {yr} {name}: {len(recs)} partners")
    with open(OUT / "demand_panel_long.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["year", "product_hs", "product", "partner_code",
                    "partner_name", "tons", "value_usd"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT/'demand_panel_long.csv'}")

    # sanity check: flag slices that look truncated (far fewer partners than the
    # product's typical count) -- usually a preview row-cap artdefact; re-run, or
    # set an API_KEY for the full data endpoint.
    import statistics
    from collections import defaultdict
    cnt = defaultdict(dict)
    for yr, hs, name, *_ in rows:
        cnt[name][yr] = cnt[name].get(yr, 0) + 1
    print("\nsanity 