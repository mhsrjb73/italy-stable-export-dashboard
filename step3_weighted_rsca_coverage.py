import os
import re
import pandas as pd
from io import StringIO

print("STEP3_WEIGHTED_RSCA_COVERAGE = START")

BASE_DIR = os.path.expanduser("~/Downloads/italy")

STABLE_FILE = os.path.join(
    BASE_DIR, "italy_hs6_stable_min3years_avg_rsca.csv"
)

PARTNER_FILES = {
    "Germany": "italy to germany.xls",
    "France": "italy to france.xls",
    "Spain": "italy to spain.xls",
    "Switzerland": "italy to switzerland.xls",
    "Poland": "italy to poland.xls",
    "Belgium": "_Italy_and_Belgium.xls",
    "Netherlands": "Italy_and_Netherlands.xls",
    "Austria": "Italy_and_Austria.xls",
    "Romania": "Italy_and_Romania.xls",
    "Czech Republic": "Italy_and_Czech_Republic .xls",
}

YEAR_MIN, YEAR_MAX = 2013, 2024
YEARS = list(range(YEAR_MIN, YEAR_MAX + 1))

# ---------------- utils ----------------
def normalize_hs6(s):
    return (
        s.astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )

def to_number(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace(" ", "").replace(",", "")
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s in {"", "-", "."}:
        return 0.0
    try:
        return float(s)
    except:
        return 0.0

def read_trademap_html(path):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                html = f.read()
            tables = pd.read_html(StringIO(html))
            if tables:
                return max(tables, key=lambda x: x.shape[0])
        except:
            continue
    raise RuntimeError(f"Cannot read TradeMap file: {path}")

def fix_headers(df):
    if df.shape[0] < 3:
        return df
    if "product code" not in str(df.iloc[0, 0]).lower():
        return df

    top = df.iloc[0].astype(str)
    sub = df.iloc[1].astype(str)
    cols = []
    for t, s in zip(top, sub):
        if s.lower() in {"nan", ""}:
            cols.append(t)
        else:
            cols.append(f"{t} | {s}")
    out = df.iloc[2:].copy()
    out.columns = cols
    return out.reset_index(drop=True)

def find_hs_col(df):
    for c in df.columns:
        if str(c).startswith("Product code"):
            return c
    # fallback
    scores = {}
    for c in df.columns:
        v = normalize_hs6(df[c])
        scores[c] = v.str.match(r"\d{6}").sum()
    return max(scores, key=scores.get)

def partner_year_cols(df, partner):
    res = {}
    prefix = f"Italy's exports to {partner}".lower()
    for c in df.columns:
        name = str(c).lower()
        if prefix in name and "value in" in name:
            for y in YEARS:
                if str(y) in name:
                    res[y] = c
    return res

# ---------------- load RSCA ----------------
stable = pd.read_csv(STABLE_FILE)
stable["hs6"] = normalize_hs6(stable["hs6"])
stable = stable[stable["hs6"] != "000000"]

RSCA_MAP = dict(zip(stable["hs6"], stable["avg_rsca"]))
TOTAL_RSCA = sum(RSCA_MAP.values())

print("Stable HS6:", len(RSCA_MAP))
print("Total RSCA weight:", round(TOTAL_RSCA, 3))

results = []

# ---------------- main loop ----------------
for partner, fname in PARTNER_FILES.items():
    print("Processing:", partner)

    df = read_trademap_html(os.path.join(BASE_DIR, fname))
    df = fix_headers(df)

    hs_col = find_hs_col(df)
    df["hs6"] = normalize_hs6(df[hs_col])
    df = df[df["hs6"].isin(RSCA_MAP)]

    ycols = partner_year_cols(df, partner)
    if not ycols:
        raise ValueError(f"No year columns found for {partner}")

    vals = pd.DataFrame({
        y: df[ycols[y]].map(to_number)
        for y in ycols
    })

    exported = (vals > 0).any(axis=1)
    exported_hs = set(df.loc[exported, "hs6"])

    weighted_sum = sum(RSCA_MAP[h] for h in exported_hs)

    results.append({
        "partner": partner,
        "exported_stable_hs6": len(exported_hs),
        "weighted_rsca_sum": weighted_sum,
        "weighted_rsca_coverage": weighted_sum / TOTAL_RSCA
    })

# ---------------- save ----------------
out = pd.DataFrame(results).sort_values(
    "weighted_rsca_coverage", ascending=False
)

out_path = os.path.join(
    BASE_DIR, "step3_partner_weighted_rsca_coverage.csv"
)
out.to_csv(out_path, index=False)

print("DONE ✔")
print(out)
print("Saved:", out_path)

