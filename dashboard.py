import os
import re
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Italy Stable Export Advantage Dashboard", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def pick_partner_col(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns]
    # common candidates
    for cand in ["partner", "Partner", "country", "Country", "p", "name", "Name"]:
        if cand in cols:
            return cand
    # fallback: any col containing 'partner' or 'country'
    for c in cols:
        lc = c.lower()
        if "partner" in lc or "country" in lc:
            return c
    raise KeyError(f"No partner/country column found. Columns: {cols}")

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    return normalize_cols(df)

def ensure_partner(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    pcol = pick_partner_col(df)
    if pcol != "partner":
        df = df.rename(columns={pcol: "partner"})
    df["partner"] = df["partner"].astype(str).str.strip()
    return df

def pick_numeric_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try contains
    for c in df.columns:
        lc = c.lower()
        for pat in candidates:
            if pat.lower() in lc:
                return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    step1 = ensure_partner(safe_read_csv("step1_partner_value_share_stable.csv"))
    step2 = safe_read_csv("step2_hs6_partner_frequency.csv")  # hs6-level, no partner needed
    step3 = ensure_partner(safe_read_csv("step3_partner_weighted_rsca_coverage.csv"))
    step4 = ensure_partner(safe_read_csv("step4_partner_clusters.csv"))
    return step1, step2, step3, step4

try:
    step1, step2, step3, step4 = load_data()
except Exception as e:
    st.error("Dashboard failed to load data.")
    st.exception(e)
    st.stop()

# ----------------------------
# Column mapping (robust)
# ----------------------------
col_value_share = pick_numeric_col(step1, ["value_share_stable"])
col_cov_ratio   = pick_numeric_col(step4, ["coverage_ratio"])
col_weight_cov  = pick_numeric_col(step3, ["weighted_rsca_coverage"])

# ----------------------------
# Title & Intro
# ----------------------------
st.title("🇮🇹 Italy’s Stable Export Advantage Dashboard (2013–2024)")
st.markdown(
"""
This dashboard explores Italy’s **stable comparative advantage core** (HS6 products with persistent positive RSCA)
and evaluates how strongly it is absorbed across **10 major European partners**.

**Source:** Trade Map (International Trade Centre — ITC) • **Author’s calculations**
"""
)
st.divider()

# ----------------------------
# KPIs
# ----------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Stable HS6 products (core)", int(pd.read_csv("italy_hs6_stable_min3years_avg_rsca.csv").shape[0]))
k2.metric("Partners analysed", int(step1["partner"].nunique()))
k3.metric("Max weighted RSCA coverage", f"{step3[col_weight_cov].max():.2f}")

st.divider()

# ----------------------------
# Partner selector
# ----------------------------
partners = sorted(set(step1["partner"]).intersection(set(step3["partner"])).intersection(set(step4["partner"])))
if not partners:
    st.error("No common partner names across step1/step3/step4 files. Check partner naming consistency.")
    st.stop()

partner = st.selectbox("Select a partner country", partners)

# Extract row metrics safely
p1 = step1.loc[step1["partner"] == partner].iloc[0]
p3 = step3.loc[step3["partner"] == partner].iloc[0]
p4 = step4.loc[step4["partner"] == partner].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Value share of stable HS6", f"{float(p1[col_value_share]):.2%}")
c2.metric("Coverage ratio (HS6 presence)", f"{float(p4[col_cov_ratio]):.2%}")
c3.metric("Weighted RSCA coverage", f"{float(p3[col_weight_cov]):.2%}")

st.divider()

# ----------------------------
# Charts
# ----------------------------
st.subheader("1) Value share of stable products by partner")
fig1 = px.bar(
    step1.sort_values(col_value_share, ascending=False),
    x="partner",
    y=col_value_share,
    text_auto=".2%",
    labels={col_value_share: "Value share"},
)
st.plotly_chart(fig1, use_container_width=True)
st.markdown("**Key insight:** Stable-advantage products form a *select core*, but they contribute a meaningful share of export value across partners.")

st.divider()

st.subheader("2) Coverage of stable HS6 products across partners")
fig2 = px.bar(
    step4.sort_values(col_cov_ratio, ascending=False),
    x="partner",
    y=col_cov_ratio,
    text_auto=".2%",
    labels={col_cov_ratio: "Coverage ratio"},
)
st.plotly_chart(fig2, use_container_width=True)
st.markdown("**Key insight:** Market reach is uneven—many stable advantages do not diffuse uniformly across all partners.")

st.divider()

st.subheader("3) Weighted RSCA absorption by partner")
fig3 = px.bar(
    step3.sort_values(col_weight_cov, ascending=False),
    x="partner",
    y=col_weight_cov,
    text_auto=".2%",
    labels={col_weight_cov: "Weighted RSCA coverage"},
)
st.plotly_chart(fig3, use_container_width=True)
st.markdown("**Key insight:** When product strength is weighted (RSCA), partner rankings shift—revealing structural alignment beyond volume alone.")

st.divider()

st.subheader("4) Partner typology: Value depth vs Structural alignment")
# merge minimal metrics for scatter
scatter = step4[["partner", col_cov_ratio]].merge(
    step1[["partner", col_value_share]], on="partner", how="left"
).merge(
    step3[["partner", col_weight_cov]], on="partner", how="left"
)
# add cluster if available
cluster_col = None
for cand in ["cluster", "Cluster"]:
    if cand in step4.columns:
        cluster_col = cand
        break
if cluster_col:
    scatter = scatter.merge(step4[["partner", cluster_col]], on="partner", how="left")

fig4 = px.scatter(
    scatter,
    x=col_value_share,
    y=col_weight_cov,
    color=cluster_col if cluster_col else None,
    text="partner",
    labels={
        col_value_share: "Value share of stable HS6",
        col_weight_cov: "Weighted RSCA coverage",
    },
)
fig4.update_traces(textposition="top center")
st.plotly_chart(fig4, use_container_width=True)
st.markdown("**Key insight:** Partners separate into types—some absorb high-value stable exports, while others align more with Italy’s strongest structural advantages.")

st.divider()
st.caption("Author: Mahsa Rajabi Nejad — Italy Stable RSCA Project")
