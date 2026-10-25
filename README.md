# PitNET MRI Radiomics Reproducibility & Stability Pipeline

An open-source research and computational benchmarking suite for auditing the stability, reproducibility, and translational calibration of MRI radiomic biomarkers in pituitary neuroendocrine tumors (PitNETs) under segmentation boundary uncertainty and multi-model variations.

---

## Overview

High-throughput MRI radiomics holds promise for non-invasive neuroendocrine phenotyping, preoperative consistency assessment, and proliferative risk stratification in PitNETs. However, the sellar and parasellar microenvironments present unique anatomical challenges: tumors interface directly with the enhancing cavernous sinuses, internal carotid artery flow voids, and compressed normal pituitary tissue. Subtle inter-observer or algorithmic contour variations of only 1–2 voxels can inadvertently sample hyperintense cavernous venous blood or hypointense flow voids, provoking severe volatility in higher-order texture metrics.

This codebase implements a systematic reproducibility audit across 107 Image Biomarker Standardisation Initiative (IBSI) features and 50 PitNET patients, evaluating 12 experimental conditions including manual expert consensus, controlled synthetic boundary perturbations ($\pm1$ and $\pm2$ voxel dilation/erosion, 3D elastic deformation), and 5 automated deep-learning architectures (nnU-Net, 3D CNN ensemble, 2.5D CNN, GrowCut).

Key contributions:
- **Reproducibility Audit**: Stratification into Tier 1 (Robust: $ICC \ge 0.85, CV \le 30\%$), Tier 2 (Moderate), and Tier 3 (Fragile).
- **Synthetic vs. Algorithmic Concordance**: Demonstration that controlled morphological shifts serve as a high-fidelity surrogate ($r = 0.939, p < 10^{-45}$) for multi-model deep learning variations.
- **Diagnostic Stress-Testing**: Demonstration of calibration collapse in naive unpruned radiomics (+470.7% Brier score error under erosion) versus calibration invariance (<6.5% drift) in robust-pruned pipelines.
- **Certified PitNET Robust Radiomics Atlas**: Catalog of 16 certified Tier 1 features invariant to sellar segmentation noise.

---

## Open Cohorts

| Cohort | Size | MRI Modality | Annotations & Segmentations | Metadata / Endpoints | Access / Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prague Cohort** | N=50 | 3T multiparametric (CE-T1, T1, T2, FIESTA, ADC) | Expert ground truth + 5 automated benchmark models (nnU-Net full, nnU-Net rep, 3D CNN ensemble, 2.5D CNN, GrowCut) | Sellar anatomical boundaries, benchmark segmentation metrics | OpenNeuro `ds006248` |
| **UCL Cohort** | N=136 | Isovolumetric CE-T1 (+ T2 subset) | Expert tumor & internal carotid artery contours | Presentation (NFPA, Cushing, Acromegaly, Prolactinoma), Ki-67 index, Knosp invasion | Figshare (`10.6084/m9.figshare.27894084`) |

> **Note on Study Scope and Clinical Endpoints**:
> - **Primary Benchmark (Prague Cohort, $N=50$)**: This investigation uses the Prague OpenNeuro dataset because it uniquely provides expert consensus ground-truth delineations alongside five distinct pre-computed automated deep-learning models (nnU-Nets, 3D CNNs, 2.5D CNNs, GrowCut). Downstream machine learning stress testing uses an objective composite phenotypic burden target (median volume and entropy) strictly as an operational sensitivity probe to measure probability calibration collapse (Brier score surge) under spatial boundary noise.
> - **Prospective Clinical Translation (UCL Cohort, $N=136$)**: Cohort pipelines and endpoint interfaces for Ki-67 proliferation, Knosp invasion, and hormone hypersecretion are provided in `code/` to support future multi-center validation applying the certified 16-feature Robust Atlas to true clinical endpoints.

---

## Directory Layout

```
.
├── code/                      # Python source code for data loading, perturbations, and extraction
│   ├── 01_dataset_loader.py
│   ├── 02_perturbation_engine.py
│   ├── 03_radiomic_extraction.py
│   ├── 04_icc_reproducibility_audit.py
│   ├── 05_clinical_stress_testing.py
│   ├── 06_generate_figures_and_tables.py
│   ├── run_pipeline.py        # Master pipeline CLI orchestrator
│   ├── config.yaml            # Pipeline hyperparameters and extraction settings
│   ├── requirements.txt       # Python package dependencies
│   └── README.md              # Detailed pipeline documentation
├── analysis/                  # Aggregate results, feature matrices, figures, and tables
│   ├── results/
│   │   ├── features/          # Extracted master radiomics matrices
│   │   └── metrics/           # Feature stability audit and stress-testing summaries
│   ├── figures/               # High-resolution publication figures (Figures 1-5)
│   └── tables/                # LaTeX tables (Tables 1-3)
├── docs/                      # Scientific documentation and study design
│   ├── RESEARCH_BRIEF.md
│   ├── PROBLEM_ANCHOR.md
│   ├── DATA_VERIFICATION.md
│   └── STUDY_DESIGN.md
└── manuscript/                # LaTeX manuscript sources and compiled PDF
```

---

## Quickstart

1. **Environment Setup**:
   ```bash
   pip install -r code/requirements.txt
   ```

2. **Execute Full Pipeline**:
   ```bash
   python code/run_pipeline.py --step all
   ```

3. **Stage-by-Stage Execution**:
   ```bash
   python code/run_pipeline.py --step audit_data
   python code/run_pipeline.py --step perturb
   python code/run_pipeline.py --step extract
   python code/run_pipeline.py --step icc_audit
   python code/run_pipeline.py --step stress_test
   python code/run_pipeline.py --step export_tables
   ```

---

## License

This project is released under the MIT Open Source License.
