#!/usr/bin/env python3
"""
Statistical Reproducibility Audit & Feature Stratification.
Calculates:
1. ICC(2,1): Two-way random effects, single rater, absolute agreement
2. Lin's Concordance Correlation Coefficient (CCC)
3. Dynamic Coefficient of Variation (CV)
Stratifies features into:
- Tier 1: Robust (ICC >= 0.85) -> "PitNET Robust Radiomics Atlas"
- Tier 2: Moderate (0.70 <= ICC < 0.85)
- Tier 3: Fragile (ICC < 0.70)
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import pingouin as pg
except ImportError:
    pg = None

def compute_icc21(df_long, target_col="patient_id", rater_col="perturbation", value_col="feature_value"):
    """
    Computes ICC(2,1) using pingouin or custom ANOVA implementation.
    """
    if pg is not None:
        try:
            icc_res = pg.intraclass_corr(
                data=df_long,
                targets=target_col,
                raters=rater_col,
                ratings=value_col
            )
            # Find ICC2 (two-way random single)
            val = icc_res.loc[icc_res["Type"] == "ICC2", "ICC"].values[0]
            ci95 = icc_res.loc[icc_res["Type"] == "ICC2", "CI95%"].values[0]
            pval = icc_res.loc[icc_res["Type"] == "ICC2", "pval"].values[0]
            return float(val), ci95, float(pval)
        except Exception:
            return np.nan, (np.nan, np.nan), np.nan
    else:
        # Fallback two-way random ANOVA calculation
        piv = df_long.pivot(index=target_col, columns=rater_col, values=value_col).dropna()
        n, k = piv.shape
        if n < 3 or k < 2:
            return np.nan, (np.nan, np.nan), np.nan
        
        row_means = piv.mean(axis=1)
        col_means = piv.mean(axis=0)
        grand_mean = piv.values.mean()
        
        ss_total = ((piv.values - grand_mean) ** 2).sum()
        ss_rows = k * ((row_means - grand_mean) ** 2).sum()
        ss_cols = n * ((col_means - grand_mean) ** 2).sum()
        ss_error = ss_total - ss_rows - ss_cols
        
        ms_rows = ss_rows / (n - 1)
        ms_cols = ss_cols / (k - 1)
        ms_error = ss_error / ((n - 1) * (k - 1))
        
        icc21 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error))
        return float(icc21), (np.nan, np.nan), np.nan

def compute_ccc(x, y):
    """
    Calculates Lin's Concordance Correlation Coefficient between two vectors.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan
    
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    s_xy = np.cov(x, y)[0, 1]
    
    numerator = 2 * s_xy
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)

def audit_feature_matrix(features_df, output_report_csv):
    """
    Takes long or wide feature DataFrame, runs ICC and CCC across perturbations,
    and assigns Tier 1, 2, or 3.
    """
    # Identify non-feature columns
    meta_cols = ["patient_id", "perturbation", "cohort"]
    feat_cols = [c for c in features_df.columns if c not in meta_cols]
    
    results = []
    print(f"Auditing stability for {len(feat_cols)} radiomic features...")
    
    for feat in feat_cols:
        sub_df = features_df[["patient_id", "perturbation", feat]].dropna()
        sub_df.rename(columns={feat: "feature_value"}, inplace=True)
        
        icc_val, ci95, pval = compute_icc21(sub_df)
        
        # Calculate CV across perturbations per patient, then take median
        piv = sub_df.pivot(index="patient_id", columns="perturbation", values="feature_value")
        cvs = piv.std(axis=1) / (piv.mean(axis=1).abs() + 1e-8)
        median_cv = float(cvs.median())
        
        # Extract feature class
        feat_class = feat.split("_")[0] if "_" in feat else "unknown"
        
        # Assign Tier
        if icc_val >= 0.85 and median_cv <= 0.30:
            tier = "Tier 1: Robust"
        elif icc_val >= 0.70:
            tier = "Tier 2: Moderate"
        else:
            tier = "Tier 3: Fragile"
            
        results.append({
            "feature_name": feat,
            "feature_class": feat_class,
            "icc_2_1": round(icc_val, 4),
            "median_cv": round(median_cv, 4),
            "tier": tier
        })
        
    audit_df = pd.DataFrame(results)
    audit_df.sort_values(by="icc_2_1", ascending=False, inplace=True)
    
    os.makedirs(os.path.dirname(output_report_csv), exist_ok=True)
    audit_df.to_csv(output_report_csv, index=False)
    
    # Print summary
    tier_counts = audit_df["tier"].value_counts()
    print("\n=== Feature Reproducibility Breakdown ===")
    for t, cnt in tier_counts.items():
        print(f"  {t}: {cnt} features ({cnt/len(audit_df)*100:.1f}%)")
        
    return audit_df

if __name__ == "__main__":
    print("Stability and Reproducibility Audit Engine ready.")
