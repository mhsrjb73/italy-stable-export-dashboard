import os
import re
import pandas as pd
from io import StringIO

print("STEP3_WEIGHTED_RSCA_COVERAGE_V2 = START")

BASE_DIR = os.path.expanduser("~/Downloads/italy")

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

# ---------------- utils ----------------
def normalize_hs6(s: pd.Series) -> pd.Series:
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
    return df.reset_index(drop=True)

def find_hs_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip() == "Product code" or str(c).startswith("Product code |"):
            return c
    # fallback: best HS-like column
    best_col, best_score = None, -1
    for c in df.columns:
        ser = normalize_hs6(df[c])
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

def detect_rsca_column(stable: pd.DataFrame) -> str:
    # exact/likely names first
    cols_lower = {str(c).lower(): c for c in stable.columns}

    candidates = [
        "avg_rsca", "mean_rsca", "rsca_avg", "rsca_mean",
        "avg rsca", "mean rsca", "rsca (avg)", "rsca (mean)",
        "rsca", "rsca_value", "rsca score"
    ]

    for cand in candidates:
        if cand in cols_lower and cols_lower[cand] != "hs6":
            return cols_lower[cand]

    # fallback: any column containing rsca (excluding count/year/hs6)
    for c in stable.columns:
        name = str(c).lower()
        if ("rsca" in name) and ("count" not in name) and ("year" not in name) and (name != "hs6"):
            return c

    raise ValueError(f"Could not detect RSCA column. Available columns: {list(stable.columns)}")

# ---------------- load stable & RSCA ----------------
stable = pd.read_csv(STABLE_FILE)

if "hs6" not in stable.columns:
    raise ValueError("Stable file must contain column 'hs6'.")

stable["hs6"] = normalize_hs6(stable["hs6"])
stable = stable[(stable["hs6"].str.match(r"^\d{6}$")) & (stable["hs6"] != "000000")].copy()

rsca_col = detect_rsca_column(stable)
stable[rsca_col] = pd.to_numeric(stable[rsca_col], errors="coerce").fillna(0.0)

RSCA_MAP = dict(zip(stable["hs6"], stable[rsca_col]))
TOTAL_RSCA = float(sum(RSCA_MAP.values()))

print("Detected RSCA column:", rsca_col)
print("Stable HS6:", len(RSCA_MAP))
print("Total RSCA weight:", round(TOTAL_RSCA, 6))

if TOTAL_RSCA <= 0:
    raise ValueError("Total RSCA weight is <= 0. RSCA column might be wrong or empty.")

# ---------------- main loop ----------------
results = []

for partner, fname in PARTNER_FILES.items():
    print("Processing:", partner)
    path = os.path.join(BASE_DIR, fname)

    raw = read_trademap_html_main_table(path)
    df = fix_header_two_rows(raw)

    hs_col = find_hs_col(df)
    df["hs6"] = normalize_hs6(df[hs_col])

    # keep only stable HS6
    df = df[df["hs6"].isin(RSCA_MAP)].copy()
    if df.empty:
        results.append({
            "partner": partner,
            "exported_stable_hs6": 0,
            "weighted_rsca_sum": 0.0,
            "weighted_rsca_coverage": 0.0,
            "years_detected": 0
        })
        continue

    ycols = partner_value_cols(df, partner)
    if not ycols:
        raise ValueError(f"No partner year columns detected for {partner} in {fname}")

    available_years = sorted(ycols.keys())
    vals = pd.DataFrame({y: df[ycols[y]].map(to_number) for y in available_years})

    exported_mask = (vals > 0).any(axis=1)
    exported_hs = set(df.loc[exported_mask, "hs6"].unique())

    weighted_sum = float(sum(RSCA_MAP[h] for h in exported_hs))

    results.append({
        "partner": partner,
        "exported_stable_hs6": int(len(exported_hs)),
        "weighted_rsca_sum": weighted_sum,
        "weighted_rsca_coverage": weighted_sum / TOTAL_RSCA,
        "years_detected": int(len(available_years))
    })

out = pd.DataFrame(results).sort_values("weighted_rsca_coverage", ascending=False)

out_path = os.path.join(BASE_DIR, "step3_partner_weighted_rsca_coverage.csv")
out.to_csv(out_path, index=False)

print("DONE ✔")
print(out.to_string(index=False))
print("Saved:", out_path)

