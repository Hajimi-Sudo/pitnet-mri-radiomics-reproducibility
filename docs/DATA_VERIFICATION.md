# Data Verification: PitNET Open MRI Cohorts

## 1. Cohort 1: UCL Pituitary Neuroendocrine Tumors (Pandit et al., 2025)

- **Source**: Scientific Data, Nature Portfolio (DOI: `10.1038/s41597-024-04218-8`, PMID: `39814754`)
- **Repository**: Figshare (DOI: `10.6084/m9.figshare.27894084`)
- **Cohort Size**: N = 136 adult patients undergoing transsphenoidal surgery between July 2021 and December 2023.
- **Inclusion Criteria**:
  - Histopathologically verified PitNET (adenoma).
  - High spatial resolution pre-operative contrast-enhanced T1-weighted (CE-T1) brain MRI.
  - Multi-disciplinary tumor board (MDT) consensus and patient consent.
- **File Structure per Patient**:
  - `[pid]_T1.nii.gz`: Defaced, pre-processed isovolumetric CE-T1 MRI.
  - `[pid]_T2.nii.gz`: Co-registered T2-weighted MRI (present in patient subset).
  - `[pid]_tumour.nii.gz`: Expert tumor segmentation via ITK-SNAP (region-competitive active contour, balloon 0.1, curvature 0.9).
  - `[pid]_carotids.nii.gz`: Segmented bilateral internal carotid arteries.
- **Clinical & Endocrine Metadata**:
  - `age`, `sex`
  - `clinical_phenotype`: Non-functioning pituitary adenoma (NFPA), Cushing's disease, Acromegaly, Prolactinoma.
  - `who_classification`: WHO 2022 CNS tumor lineage (PIT1, TPIT, SF1, null cell).
  - `ki67_index`: Histopathological proliferation index (percentage).
  - `knosp_invasion`: Carotid encasement and parasellar invasion status.

---

## 2. Cohort 2: OpenNeuro Multimodal Pituitary Adenoma (Černý et al., 2025)

- **Source**: OpenNeuro Dataset `ds006248` (DOI: `10.18112/openneuro.ds006248.v1.0.0`)
- **Institution**: Department of Neurosurgery and Neurooncology, Military University Hospital Prague, Charles University.
- **Cohort Size**: N = 50 patients scanned on a uniform 3T GE Discovery 750w MR scanner.
- **Sequences**:
  - Coronal CE-T1 (100% of patients)
  - Coronal T1 non-enhanced (100%)
  - Sagittal CE-T1 (98%)
  - Coronal T2 (86%)
  - 3D Axial T1+C (98%)
  - Coronal FIESTA (68%) / CE-FIESTA (70%)
  - Axial ADC b=200/b=1000 (56-60%)
- **Multi-Class Ground Truth Segmentations (`/derivatives/`)**:
  - Single-label integer NIfTI files:
    - `0`: Background
    - `1`: Tumor
    - `2`: Normal pituitary gland
    - `3`: Left ICA
    - `4`: Right ICA
    - `5`: Optic pathway
- **Pre-computed Segmentation Benchmark Models**:
  - `cerny2025`: nnU-Net full volume (mean Dice 0.815 ± 0.110)
  - `damutten2025`: nnU-Net full volume (mean Dice 0.794 ± 0.084)
  - `damutten2024`: 3D CNN ensemble (mean Dice 0.667 ± 0.137)
  - `cerny2023`: 2.5D keyframe CNN (mean Dice 0.779 ± 0.233)
  - `egger2012`: Semi-automated GrowCut (mean Dice 0.730 ± 0.124)

---

## 3. Data Integration & Harmonization Strategy

1. **Volume Extraction**:
   - For Cohort 1, extract binary tumor mask from `[pid]_tumour.nii.gz`.
   - For Cohort 2, isolate label `1` (tumor) from multi-class ground truth and each benchmark model.
2. **Spatial Resampling**:
   - Resample all MRI volumes and binary masks to an isotropic voxel grid ($1.0 \times 1.0 \times 1.0\text{ mm}^3$) using B-spline interpolation for images and nearest-neighbor interpolation for masks.
3. **Intensity Standardization**:
   - Bias field correction (N4ITK).
   - Z-score normalization of brain parenchyma or sellar region with clipping at $\pm 3\sigma$.
