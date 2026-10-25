#!/usr/bin/env python3
"""
Clinical & Diagnostic Stress-Testing Engine: Downstream Model Robustness & Calibration Audit.
Quantifies the diagnostic and calibration penalty of radiomic feature instability under spatial boundary uncertainty.

Methodology:
- Primary Benchmark Cohort: Prague OpenNeuro ds006248 (N=50 PitNET patients).
- Operational Composite Phenotypic Target: High vs. Low tumor volume/burden (stratified by median tumor volume),
  serving strictly as a standardized sensitivity probe to evaluate discrimination (AUC) and
  probability calibration drift (Brier score loss) under spatial delineation uncertainty.
- Experimental Design:
  1. Models trained exclusively on baseline ground-truth expert segmentations using 5-fold stratified cross-validation.
  2. Features standardized using z-score normalization computed strictly within each training fold.
  3. L1-penalized LASSO logistic regression head (SAGA solver, max_iter=2000, random_state=42).
     By default, inverse regularization strength is fixed to C=0.5 (corresponding to sparsity penalty
     lambda = 1 / (C * N_train)), maintaining consistent sparsity across folds without sub-fold variance.
     An optional nested cross-validation mode (use_nested_cv=True) is also supported via LogisticRegressionCV.
  4. Models deployed at inference without retraining across 7 test perturbation regimes (synthetic dilation/erosion/elastic
     shifts and real-world deep-learning segmentations).

Pipelines Evaluated:
- Naive Unpruned Pipeline: All 107 IBSI features -> L1-penalized LASSO selection -> Logistic Classifier
- Robust-Pruned Pipeline: Restricted to 16 Certified Tier 1 Features (ICC >= 0.85, CV <= 30%) -> L1 selection -> Classifier
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, brier_score_loss


def run_stress_test(
    features_all_perts_df,
    robust_features_list,
    target_col="target",
    n_splits=5,
    random_state=42,
    C_fixed=0.5,
    use_nested_cv=False,
    pert_order=None,
    output_summary_csv=None
):
    """
    Executes 5-fold cross-validation stress test across all perturbation states.
    """
    baseline_df = features_all_perts_df[features_all_perts_df["perturbation"] == "baseline_gt"].copy()
    baseline_df = baseline_df.sort_values("patient_id").reset_index(drop=True)

    meta_cols = ["patient_id", "sequence", "perturbation", "cohort", target_col]
    all_feature_names = [c for c in baseline_df.columns if c not in meta_cols]
    valid_robust_cols = [f for f in robust_features_list if f in all_feature_names]

    y_baseline = baseline_df[target_col].astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if pert_order is None:
        pert_order = [
            "baseline_gt",
            "dilate_1vox",
            "dilate_2vox",
            "elastic_s101",
            "erode_1vox",
            "erode_2vox",
            "pred_cerny2025",
            "pred_damutten2024"
        ]

    # Filter to existing perturbations
    available_perts = features_all_perts_df["perturbation"].unique()
    pert_order = [p for p in pert_order if p in available_perts]

    results = []

    for pert in pert_order:
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(baseline_df, y_baseline)):
            train_df = baseline_df.iloc[train_idx]
            test_pids = baseline_df.iloc[test_idx]["patient_id"].values
            y_train = train_df[target_col].values

            # Scale Pipeline A (Naive: all features)
            sc_A = StandardScaler()
            X_tr_A = train_df[all_feature_names].values
            X_tr_A_s = sc_A.fit_transform(X_tr_A)

            # Scale Pipeline B (Robust: Tier 1 features only)
            sc_B = StandardScaler()
            X_tr_B = train_df[valid_robust_cols].values
            X_tr_B_s = sc_B.fit_transform(X_tr_B)

            # Fit Classifiers
            if use_nested_cv:
                clf_A = LogisticRegressionCV(
                    Cs=[0.1, 0.2, 0.5, 1.0],
                    cv=3,
                    penalty="l1",
                    solver="saga",
                    max_iter=2000,
                    random_state=random_state
                )
                clf_B = LogisticRegressionCV(
                    Cs=[0.1, 0.2, 0.5, 1.0],
                    cv=3,
                    penalty="l1",
                    solver="saga",
                    max_iter=2000,
                    random_state=random_state
                )
            else:
                clf_A = LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=C_fixed,
                    max_iter=2000,
                    random_state=random_state
                )
                clf_B = LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=C_fixed,
                    max_iter=2000,
                    random_state=random_state
                )

            clf_A.fit(X_tr_A_s, y_train)
            clf_B.fit(X_tr_B_s, y_train)

            # Test evaluation on the specific perturbation state
            sub_pert = features_all_perts_df[
                (features_all_perts_df["perturbation"] == pert) &
                (features_all_perts_df["patient_id"].isin(test_pids))
            ].copy()
            if target_col not in sub_pert.columns:
                sub_pert = sub_pert.merge(baseline_df[["patient_id", target_col]], on="patient_id")

            y_test = sub_pert[target_col].values
            X_te_A = sub_pert[all_feature_names].values
            X_te_A_s = sc_A.transform(X_te_A)
            probs_A = clf_A.predict_proba(X_te_A_s)[:, 1]

            X_te_B = sub_pert[valid_robust_cols].values
            X_te_B_s = sc_B.transform(X_te_B)
            probs_B = clf_B.predict_proba(X_te_B_s)[:, 1]

            auc_A = roc_auc_score(y_test, probs_A) if len(np.unique(y_test)) > 1 else np.nan
            brier_A = brier_score_loss(y_test, probs_A)
            auc_B = roc_auc_score(y_test, probs_B) if len(np.unique(y_test)) > 1 else np.nan
            brier_B = brier_score_loss(y_test, probs_B)

            fold_metrics.append({
                "fold": fold,
                "auc_naive": auc_A,
                "brier_naive": brier_A,
                "auc_robust": auc_B,
                "brier_robust": brier_B
            })

        m = pd.DataFrame(fold_metrics).mean()
        results.append({
            "perturbation": pert,
            "auc_naive": round(float(m["auc_naive"]), 3),
            "auc_robust": round(float(m["auc_robust"]), 3),
            "brier_naive": round(float(m["brier_naive"]), 3),
            "brier_robust": round(float(m["brier_robust"]), 3)
        })

    summary_df = pd.DataFrame(results)
    print("\n=== Downstream Classifier Stress-Testing Summary ===")
    print(summary_df.to_string(index=False))

    if output_summary_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_summary_csv)), exist_ok=True)
        summary_df.to_csv(output_summary_csv, index=False)
        print(f"[Export] Summary metrics saved to: {output_summary_csv}")

    return summary_df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))

    features_csv = os.path.join(root_dir, "analysis/results/features/master_radiomics_features.csv")
    audit_csv = os.path.join(root_dir, "analysis/results/metrics/feature_stability_audit.csv")
    output_csv = os.path.join(root_dir, "analysis/results/metrics/clinical_stress_test_summary.csv")

    if not os.path.exists(features_csv):
        print(f"[Error] Features file not found: {features_csv}")
        sys.exit(1)

    print(f"[Loading] Reading features from: {features_csv}")
    feat_df = pd.read_csv(features_csv)

    # Derive composite phenotypic target: median tumor volume split
    baseline_sub = feat_df[feat_df["perturbation"] == "baseline_gt"].copy()
    vol = baseline_sub["original_shape_MeshVolume"]
    med_vol = float(vol.median())
    print(f"[Target] Defining phenotypic tumor burden: Median volume = {med_vol:.2f} mm^3")
    
    # Assign target to all patient records
    pid_to_target = {row["patient_id"]: int(row["original_shape_MeshVolume"] >= med_vol) for _, row in baseline_sub.iterrows()}
    feat_df["target"] = feat_df["patient_id"].map(pid_to_target)

    # Load Tier 1 robust features
    if os.path.exists(audit_csv):
        audit_df = pd.read_csv(audit_csv)
        tier1_list = audit_df[audit_df["tier"].str.contains("Tier 1", case=False, na=False)]["feature"].tolist()
        print(f"[Audit] Loaded {len(tier1_list)} Tier 1 robust features.")
    else:
        print("[Warning] Audit CSV not found, fallback to default Tier 1 list.")
        tier1_list = []

    # Run stress test benchmark
    run_stress_test(
        features_all_perts_df=feat_df,
        robust_features_list=tier1_list,
        target_col="target",
        n_splits=5,
        random_state=42,
        C_fixed=0.5,
        use_nested_cv=False,
        output_summary_csv=output_csv
    )
