#!/usr/bin/env python3
"""
Clinical Stress-Testing Engine: Downstream Model Robustness & Probability Calibration Audit.
Quantifies the diagnostic and calibration penalty of radiomic feature instability under spatial boundary uncertainty.

Primary Benchmark Task (Prague OpenNeuro ds006248 cohort, N=50):
- Operational Composite Phenotypic Target: High vs. Low composite tumor burden & structural complexity
  (stratified by joint median tumor volume and first-order entropy).
  This serves strictly as a simulation sensitivity probe to benchmark probability calibration drift
  (Brier score explosion) and discriminative stability (AUC) under spatial delineation noise.

External Clinical Tasks Interface (for prospective multi-center cohorts, e.g., UCL Sci Data 2025, N=136):
- Endocrine Functionality (GH/ACTH/PRL vs. NFPA)
- Proliferative Aggressiveness (Ki-67 >= 3% vs. < 3%)
- Cavernous Sinus Invasion (Knosp Grade 3-4 vs. 0-2)

Pipelines Compared:
- Pipeline A (Naive): All 107 IBSI features -> L1-penalized LASSO selection -> Logistic Classifier
- Pipeline B (Robust-Pruned): Tier 1 Robust Features (ICC >= 0.85, CV <= 30%) -> L1 selection -> Logistic Classifier
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

def run_stress_test(
    features_all_perts_df,
    clinical_df,
    target_col,
    robust_features_list,
    n_splits=5,
    random_state=42
):
    """
    Trains on baseline masks, then tests across baseline AND all perturbed test masks.
    """
    # Merge clinical target with baseline features
    baseline_df = features_all_perts_df[features_all_perts_df["perturbation"] == "baseline"]
    merged = pd.merge(baseline_df, clinical_df[["patient_id", target_col]], on="patient_id").dropna(subset=[target_col])
    
    y = merged[target_col].astype(int).values
    all_feature_names = [c for c in baseline_df.columns if c not in ["patient_id", "perturbation", "cohort", target_col]]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    results = []
    perturbation_types = features_all_perts_df["perturbation"].unique()
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(merged, y)):
        train_pids = merged.iloc[train_idx]["patient_id"].values
        test_pids = merged.iloc[test_idx]["patient_id"].values
        
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # 1. Pipeline A: Naive (All features)
        X_train_A = merged.iloc[train_idx][all_feature_names].values
        scaler_A = StandardScaler()
        X_train_A_scaled = scaler_A.fit_transform(X_train_A)
        
        clf_A = LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=2000, random_state=random_state)
        clf_A.fit(X_train_A_scaled, y_train)
        
        # 2. Pipeline B: Robust-Pruned (Only Tier 1 features)
        valid_robust_cols = [f for f in robust_features_list if f in all_feature_names]
        X_train_B = merged.iloc[train_idx][valid_robust_cols].values
        scaler_B = StandardScaler()
        X_train_B_scaled = scaler_B.fit_transform(X_train_B)
        
        clf_B = LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=2000, random_state=random_state)
        clf_B.fit(X_train_B_scaled, y_train)
        
        # Evaluate on each perturbation in test fold
        for pert in perturbation_types:
            test_pert_df = features_all_perts_df[
                (features_all_perts_df["perturbation"] == pert) &
                (features_all_perts_df["patient_id"].isin(test_pids))
            ]
            # Ensure correct patient alignment
            test_pert_df = test_pert_df.set_index("patient_id").reindex(test_pids).reset_index()
            
            # Predict Pipeline A
            X_test_A = test_pert_df[all_feature_names].values
            X_test_A_scaled = scaler_A.transform(X_test_A)
            probs_A = clf_A.predict_proba(X_test_A_scaled)[:, 1]
            auc_A = roc_auc_score(y_test, probs_A) if len(np.unique(y_test)) > 1 else np.nan
            brier_A = brier_score_loss(y_test, probs_A)
            
            # Predict Pipeline B
            X_test_B = test_pert_df[valid_robust_cols].values
            X_test_B_scaled = scaler_B.transform(X_test_B)
            probs_B = clf_B.predict_proba(X_test_B_scaled)[:, 1]
            auc_B = roc_auc_score(y_test, probs_B) if len(np.unique(y_test)) > 1 else np.nan
            brier_B = brier_score_loss(y_test, probs_B)
            
            results.append({
                "fold": fold,
                "target": target_col,
                "perturbation": pert,
                "auc_naive": auc_A,
                "brier_naive": brier_A,
                "auc_robust": auc_B,
                "brier_robust": brier_B,
                "delta_auc": auc_B - auc_A
            })
            
    res_df = pd.DataFrame(results)
    summary = res_df.groupby("perturbation")[["auc_naive", "auc_robust", "brier_naive", "brier_robust"]].mean()
    print(f"\n=== Stress-Testing Performance Summary for [{target_col}] ===")
    print(summary.round(3))
    return res_df

if __name__ == "__main__":
    print("Clinical Stress-Testing Engine ready.")
