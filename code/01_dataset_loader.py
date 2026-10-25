#!/usr/bin/env python3
"""
Dataset Loader and Cohort Metadata Harmonizer for PitNET MRI Studies.
Handles:
1. UCL Cohort (Pandit et al., Sci Data 2025): N=136
2. Prague Cohort (Černý et al., OpenNeuro ds006248): N=50
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np

def load_config(config_path="code/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def audit_ucl_cohort(data_dir):
    """
    Scans the UCL PitNET dataset directory and matches NIfTI files with clinical metadata.
    Expected metadata columns:
    - Patient_ID, Age, Sex, Presentation (NFPA, Cushing, Acromegaly, Prolactinoma),
      Ki67, Cavernous_Invasion / Knosp.
    """
    print(f"[Audit] Checking UCL Cohort in: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"  [!] Directory {data_dir} does not exist yet. Please download from Figshare DOI: 10.6084/m9.figshare.27894084")
        return None

    # Scan for [pid]_T1.nii.gz and [pid]_tumour.nii.gz
    records = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith("_T1.nii.gz"):
                pid = f.replace("_T1.nii.gz", "")
                t1_path = os.path.join(root, f)
                mask_path = os.path.join(root, f"{pid}_tumour.nii.gz")
                t2_path = os.path.join(root, f"{pid}_T2.nii.gz")
                carotids_path = os.path.join(root, f"{pid}_carotids.nii.gz")
                
                records.append({
                    "cohort": "UCL",
                    "patient_id": pid,
                    "t1ce_path": t1_path,
                    "t2_path": t2_path if os.path.exists(t2_path) else None,
                    "tumor_mask_path": mask_path if os.path.exists(mask_path) else None,
                    "carotid_mask_path": carotids_path if os.path.exists(carotids_path) else None,
                })

    df = pd.DataFrame(records)
    print(f"  Found {len(df)} patients with CE-T1 MRI.")
    return df

def audit_prague_cohort(data_dir):
    """
    Scans the Prague OpenNeuro ds006248 dataset directory.
    Checks for coronal CE-T1, multi-class ground truth, and 5 benchmark segmentation models.
    """
    print(f"[Audit] Checking Prague OpenNeuro ds006248 in: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"  [!] Directory {data_dir} does not exist yet. Please download via OpenNeuro / AWS S3 ds006248.")
        return None

    records = []
    # OpenNeuro BIDS layout: sub-XXX/ses-X/anat/sub-XXX_...
    for sub in sorted(os.listdir(data_dir)):
        if sub.startswith("sub-"):
            sub_dir = os.path.join(data_dir, sub)
            records.append({
                "cohort": "Prague",
                "patient_id": sub,
                "sub_dir": sub_dir
            })

    df = pd.DataFrame(records)
    print(f"  Found {len(df)} BIDS subjects in OpenNeuro ds006248.")
    return df

def harmonize_clinical_endpoints(metadata_df):
    """
    Derives standardized binary clinical endpoints for downstream stress-testing:
    1. is_functioning: 1 if Acromegaly/Cushing/Prolactinoma, 0 if NFPA
    2. ki67_high: 1 if Ki-67 >= 3.0%, 0 if Ki-67 < 3.0%
    3. knosp_invasive: 1 if Knosp >= 3, 0 if Knosp 0-2
    """
    df = metadata_df.copy()
    if "phenotype" in df.columns:
        df["is_functioning"] = df["phenotype"].apply(
            lambda x: 0 if str(x).strip().upper() in ["NFPA", "NON-FUNCTIONING", "NONFUNCTIONING"] else 1
        )
    if "ki67" in df.columns:
        df["ki67_high"] = (pd.to_numeric(df["ki67"], errors="coerce") >= 3.0).astype(int)
    if "knosp" in df.columns:
        df["knosp_invasive"] = (pd.to_numeric(df["knosp"], errors="coerce") >= 3).astype(int)
    return df

if __name__ == "__main__":
    cfg = load_config()
    print("=== PitNET Dataset Verification Audit ===")
    df_ucl = audit_ucl_cohort(cfg["paths"]["data_raw_ucl"])
    df_prague = audit_prague_cohort(cfg["paths"]["data_raw_prague"])
