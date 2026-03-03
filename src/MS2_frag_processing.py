# ============================================================
# MS2_frag_Processing.py
# Combine: (1) match WS_data to MassBankEU by DTXSID
#          (2) drop missing FRAGMENTS/SMILES
#          (3) parse FRAGMENTS into frag_list
#          (4) drop rows with empty frag_list
#          (5) remove duplicates (SMILES, FRAGMENTS, frag_list)
# ============================================================

import pandas as pd
import numpy as np

# ----------------------------
# USER PATHS
# ----------------------------
WS_PATH   = "/content/WS_data.csv"
MBEU_PATH = "/content/MassBankEU_CmpdsV1.csv"

OUT_MATCHED            = "/content/matching_dtxsid_rows.csv"
OUT_CLEANED            = "/content/cleaned_fragments.csv"
OUT_PARSED             = "/content/fragments_cleaned_and_parsed.csv"
OUT_FINAL_UNIQUE       = "/content/ms2_fragments_final_unique.csv"

# ----------------------------
# Helpers
# ----------------------------
def parse_fragments(frag):
    """
    Parse fragment string into a list of floats (rounded to 3 dp).
    Accepts separators: ':', ';', '|', ','.
    Returns [] if malformed or missing.
    """
    if pd.isna(frag):
        return []
    frag = str(frag).strip()
    if frag == "":
        return []

    # standardize separators to ':'
    for sep in [';', '|', ',']:
        frag = frag.replace(sep, ':')

    out = []
    for token in frag.split(':'):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(round(float(token), 3))
        except ValueError:
            return []  # if any token is malformed, drop the row
    return out


def standardize_missing(series: pd.Series) -> pd.Series:
    """Convert common missing-like strings to NA and strip whitespace."""
    s = series.astype(str).str.strip()
    s = s.replace(['', ' ', 'None', 'none', 'nan', 'NaN', 'NULL', 'null'], pd.NA)
    return s

# ----------------------------
# 1) Load
# ----------------------------
df1 = pd.read_csv(WS_PATH)
df2 = pd.read_csv(MBEU_PATH)

# ----------------------------
# 2) Match by DTXSID (inner join)
# ----------------------------
if "dsstox_substance_id" not in df1.columns:
    raise ValueError("WS_data.csv must contain column: dsstox_substance_id")
if "DTXSID" not in df2.columns:
    raise ValueError("MassBankEU_CmpdsV1.csv must contain column: DTXSID")

df1["dsstox_substance_id"] = df1["dsstox_substance_id"].astype(str).str.strip()
df2["DTXSID"]              = df2["DTXSID"].astype(str).str.strip()

matched = pd.merge(
    df1,
    df2,
    left_on="dsstox_substance_id",
    right_on="DTXSID",
    how="inner"
)

print(f"[Step 2] Matched rows (inner join): {len(matched)}")
matched.to_csv(OUT_MATCHED, index=False)
print(f"[Saved] {OUT_MATCHED}")

# ----------------------------
# 3) Clean missing FRAGMENTS / SMILES
# ----------------------------
if "FRAGMENTS" not in matched.columns:
    raise ValueError("After merge, expected column: FRAGMENTS")
if "SMILES" not in matched.columns:
    raise ValueError("After merge, expected column: SMILES")

matched["FRAGMENTS"] = standardize_missing(matched["FRAGMENTS"])
matched["SMILES"]    = standardize_missing(matched["SMILES"])

df_cleaned = matched.dropna(subset=["FRAGMENTS", "SMILES"]).copy()

# remove rows that are whitespace after strip
df_cleaned = df_cleaned[
    (df_cleaned["FRAGMENTS"].astype(str).str.strip() != "") &
    (df_cleaned["SMILES"].astype(str).str.strip() != "")
].reset_index(drop=True)

print(f"[Step 3] Rows after dropping missing FRAGMENTS/SMILES: {len(df_cleaned)}")
df_cleaned.to_csv(OUT_CLEANED, index=False)
print(f"[Saved] {OUT_CLEANED}")

# ----------------------------
# 4) Parse fragments → frag_list; drop empty
# ----------------------------
df_cleaned["frag_list"] = df_cleaned["FRAGMENTS"].apply(parse_fragments)
df_parsed = df_cleaned[df_cleaned["frag_list"].map(len) > 0].reset_index(drop=True)

print(f"[Step 4] Rows after parsing + dropping empty frag_list: {len(df_parsed)}")
df_parsed.to_csv(OUT_PARSED, index=False)
print(f"[Saved] {OUT_PARSED}")

# ----------------------------
# 5) Remove duplicates on (SMILES, FRAGMENTS, frag_list)
# ----------------------------
# Make a stable string key for frag_list so duplicates compare cleanly
df_parsed["frag_list_str"] = df_parsed["frag_list"].apply(lambda x: ",".join(map(str, x)))

df_unique = df_parsed.drop_duplicates(
    subset=["SMILES", "FRAGMENTS", "frag_list_str"],
    keep="first"
).reset_index(drop=True)

# Keep frag_list as list, but drop helper column if you want
df_unique = df_unique.drop(columns=["frag_list_str"])

print(f"[Step 5] Rows after removing duplicates: {len(df_unique)}")
df_unique.to_csv(OUT_FINAL_UNIQUE, index=False)
print(f"[Saved] {OUT_FINAL_UNIQUE}")

print("\n✅ Done. Final file to use/deposit:")
print(f"   {OUT_FINAL_UNIQUE}")
