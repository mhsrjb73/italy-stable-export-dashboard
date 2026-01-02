import os
import re
import pandas as pd
from io import StringIO

print("STEP1_VALUE_SHARE = START")

BASE_DIR = os.path.expanduser("~/Downloads/italy")

# stable file from your previous step (2448 HS6)
STABLE_FILE = os.path.join(BASE_DIR, "italy_hs6_stable_min3years_avg_rsca.csv")

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

def normalize_hs6_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )

def to_number(x) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s == "" or s.lower() in {"-", "n/a", "na", "null"}:
        return 0.0
    s = s.replace(" ", "").replace(",", "")
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s in {"", "-", "."}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

def read_trademap_html_main_table(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                html = f.read()
            tables = pd.read_html(StringIO(html))
            if tables:
                return max(tables, key=lambda x: x.shape[0])
        except Exception:
            continue
    raise RuntimeError(f"Cannot parse HTML tables: {path}")

def fix_header_two_rows(raw: pd.DataFrame) -> pd.DataFrame:
    # If row0/row1 look like header, combine them; else return as-is
    if raw.shape[0] < 3:
        return raw

    c00 = str(raw.iloc[0, 0]).strip().lower()
    c01 = str(raw.iloc[0, 1]).strip().lower()
    if "product code" not in c00 or "product label" not in c01:
        return raw

    top = raw.iloc[0].astype(str)
    sub = raw.iloc[1].astype(str)

    new_cols = []
    for t, s in zip(top, sub):
        t = str(t).strip()
        s = str(s).strip()
        if s.lower() in {"nan", "none"} or s == "":
            new_cols.append(t)
        else:
            new_cols.append(f"{t} | {s}")

    df = raw.iloc[2:].copy()
    df.columns = new_cols
    df = df.reset_index(drop=True)
    return df

def find_hs_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip() == "Product code":
            return c
        if str(c).startswith("Product code |"):
            return c
    # fallback quality scan
    best_col, best_score = None, -1
    for c in df.columns:
        ser = normalize_hs6_series(df[c])
        valid = ser.str.match(r"^\d{6}$") & (ser != "000000")
        score = int(valid.sum())
        if score > best_score:
            best_score, best_col = score, c
    if best_col is None or best_score <= 0:
        raise ValueError(f"HS column not detected. columns={list(df.columns)}")
    return best_col

def partner_value_cols(df: pd.DataFrame, partner: str) -> dict:
    prefix = f"Italy's exports to {partner}".lower()
    year_to_col = {}
    for c in df.columns:
        name = str(c).lower()
        if prefix in name and "value in" in name:
            for y in YEARS:
                if str(y) in name:
                    year_to_col[y] = c
    return year_to_col

# -------- load stable hs6 list
stable = pd.read_csv(STABLE_FILE)
stable["hs6"] = normalize_hs6_series(stable["hs6"])
stable_hs6 = set(stable["hs6"])
print("Stable HS6 count:", len(stable_hs6))

rows = []
by_year_rows = []

for partner, fname in PARTNER_FILES.items():
    print("Processing:", partner)
    path = os.path.join(BASE_DIR, fname)

    raw = read_trademap_html_main_table(path)
    df = fix_header_two_rows(raw)

    hs_col = find_hs_col(df)
    df["hs6"] = normalize_hs6_series(df[hs_col])

    # drop TOTAL / invalid
    df = df[(df["hs6"].str.match(r"^\d{6}$")) & (df["hs6"] != "000000")].copy()

    ycols = partner_value_cols(df, partner)
    if not ycols:
        raise ValueError(f"No partner year columns detected for {partner} in {fname}")

    # build numeric values per year (available)
    vals = {}
    for y, col in ycols.items():
        vals[y] = df[col].map(to_number)
    vals_df = pd.DataFrame(vals)

    # totals (all HS)
    total_all = float(vals_df.sum(axis=0).sum())

    # totals (stable HS only)
    stable_mask = df["hs6"].isin(stable_hs6)
    total_stable = float(vals_df.loc[stable_mask].sum(axis=0).sum())

    share = (total_stable / total_all) if total_all > 0 else 0.0

    rows.append({
        "partner": partner,
        "total_export_value_all_HS_2013_2024": total_all,
        "total_export_value_stable_HS_2013_2024": total_stable,
        "value_share_stable": share,
        "hs6_count_in_file": int(df["hs6"].nunique()),
        "stable_hs6_exported_count": int(df.loc[stable_mask, "hs6"].nunique()),
        "years_detected": int(len(ycols)),
    })

    # optional: by-year shares
    for y in sorted(ycols.keys()):
        all_y = float(vals_df[y].sum())
        stable_y = float(vals_df.loc[stable_mask, y].sum())
        by_year_rows.append({
            "partner": partner,
            "year": y,
            "export_value_all": all_y,
            "export_value_stable": stable_y,
            "value_share_stable": (stable_y / all_y) if all_y > 0 else 0.0
        })

out = pd.DataFrame(rows).sort_values("value_share_stable", ascending=False)
out_path = os.path.join(BASE_DIR, "step1_partner_value_share_stable.csv")
out.to_csv(out_path, index=False)

out_year = pd.DataFrame(by_year_rows).sort_values(["partner", "year"])
out_year_path = os.path.join(BASE_DIR, "step1_partner_value_share_stable_by_year.csv")
out_year.to_csv(out_year_path, index=False)

print("DONE ✔")
print(out.to_string(index=False))
print("Saved:", out_path)
print("Saved:", out_year_path)

