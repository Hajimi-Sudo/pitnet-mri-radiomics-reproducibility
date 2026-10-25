#!/usr/bin/env python3
"""
IBSI-Standard Radiomic Feature Extraction for PitNET Reproducibility Pipeline.
Extracts 107+ standard features (Shape, 1st Order, GLCM, GLRLM, GLSZM, GLDM, NGTDM)
across baseline masks, synthetic perturbations, and deep-learning segmentation models.
"""

import os
import sys
import yaml
import pandas as pd
from tqdm import tqdm

try:
    import radiomics
    from radiomics import featureextractor
except ImportError:
    radiomics = None

def get_ibsi_extractor(bin_width=25, resample_spacing=(1.0, 1.0, 1.0)):
    """
    Constructs an IBSI-compliant PyRadiomics extractor instance.
    """
    settings = {
        'binWidth': bin_width,
        'resampledPixelSpacing': list(resample_spacing),
        'interpolator': 'sitkBSpline',
        'correctMask': True,
        'voxelArrayShift': 1000,
        'geometryTolerance': 1e-4
    }
    
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    
    # Enable standard IBSI feature classes
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('ngtdm')
    
    return extractor

def extract_features_for_patient(image_path, mask_dict, extractor, patient_id):
    """
    Extracts features across all mask perturbations for a given patient.
    mask_dict maps {perturbation_name: mask_path}
    """
    records = []
    for pert_name, mask_path in mask_dict.items():
        if not os.path.exists(mask_path):
            continue
        try:
            result = extractor.execute(image_path, mask_path)
            # Filter out pyradiomics metadata keys starting with 'diagnostics_'
            feat_dict = {
                k: float(v) for k, v in result.items() if not k.startswith("diagnostics_")
            }
            feat_dict["patient_id"] = patient_id
            feat_dict["perturbation"] = pert_name
            records.append(feat_dict)
        except Exception as e:
            print(f"[Error] Failed extraction for {patient_id} ({pert_name}): {e}")
            
    return pd.DataFrame(records)

def batch_extract(patient_manifest, output_csv, bin_width=25):
    """
    Runs multi-subject extraction across all perturbations and writes to output_csv.
    """
    if radiomics is None:
        raise RuntimeError("pyradiomics is not installed. Run 'pip install pyradiomics' first.")
        
    extractor = get_ibsi_extractor(bin_width=bin_width)
    all_dfs = []
    
    print(f"Beginning radiomics extraction for {len(patient_manifest)} patients...")
    for item in tqdm(patient_manifest):
        pid = item["patient_id"]
        img_p = item["image_path"]
        masks = item["masks"]
        df_p = extract_features_for_patient(img_p, masks, extractor, pid)
        all_dfs.append(df_p)
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    master_df.to_csv(output_csv, index=False)
    print(f"Features saved successfully to: {output_csv}")
    return master_df

if __name__ == "__main__":
    print("Radiomics extraction module ready. Execute with config or via workflow runner.")
