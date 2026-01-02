import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("STEP4_PARTNER_CLUSTERING = START")

BASE_DIR = os.path.expanduser("~/Downloads/italy")

# ---- input files from previous steps
STEP1_FILE = os.path.join(BASE_DIR, "step1_partner_value_share_stable.csv")
STEP2_FILE = os.path.join(BASE_DIR, "italy_stable_rsca_partner_coverage.csv")
STEP3_FILE = os.path.join(BASE_DIR, "step3_partner_weighted_rsca_coverage.csv")

# ---- load
s1 = pd.read_csv(STEP1_FILE)
s2 = pd.read_csv(STEP2_FILE)
s3 = pd.read_csv(STEP3_FILE)

# ---- harmonise column names
s1 = s1.rename(columns={
    "partner": "partner",
    "value_share_stable": "value_share_stable"
})[["partner", "value_share_stable"]]

s2 = s2.rename(columns={
    "partner": "partner",
    "coverage_ratio": "coverage_ratio"
})[["partner", "coverage_ratio"]]

s3 = s3.rename(columns={
    "partner": "partner",
    "weighted_rsca_coverage": "weighted_rsca_coverage"
})[["partner", "weighted_rsca_coverage"]]

# ---- merge
df = s1.merge(s2, on="partner").merge(s3, on="partner")

print("\nMerged indicators:")
print(df)

# ---- features
X = df[[
    "coverage_ratio",
    "value_share_stable",
    "weighted_rsca_coverage"
]]

# ---- standardise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
df["cluster"] = kmeans.fit_predict(X_scaled)

# ---- cluster centroids (for interpretation)
centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=X.columns
)
centroids["cluster"] = centroids.index

# ---- save outputs
out_path = os.path.join(BASE_DIR, "step4_partner_clusters.csv")
centroids_path = os.path.join(BASE_DIR, "step4_cluster_centroids.csv")

df_sorted = df.sort_values("cluster")
df_sorted.to_csv(out_path, index=False)
centroids.to_csv(centroids_path, index=False)

print("\nDONE ✔")
print("\nClustered partners:")
print(df_sorted.to_string(index=False))

print("\nCluster centroids (original scale):")
print(centroids.to_string(index=False))

print("\nSaved:")
print(out_path)
print(centroids_path)

