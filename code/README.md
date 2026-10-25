# PitNET MRI Radiomics Reproducibility & Robustness Pipeline

This repository provides an open-source, reproducible computational pipeline for evaluating the stability of MRI radiomic biomarkers in pituitary neuroendocrine tumors (PitNETs) under spatial segmentation uncertainty and multi-model variations.

## Pipeline Architecture

The pipeline consists of modular Python components adhering to the Image Biomarker Standardisation Initiative (IBSI):

1. **`01_dataset_loader.py`**:
   - Parses OpenNeuro BIDS (`ds006248`) and UCL multi-parametric MRI cohorts.
   - Harmonizes clinical endpoints: secretory lineage, proliferative status (Ki-67 labeling index), and cavernous sinus invasion (Knosp grade).

2. **`02_perturbation_engine.py`**:
   - Implements controlled 3D morphological shifts ($\pm1$ and $\pm2$ voxel dilation/erosion).
   - Generates 3D elastic B-spline boundary jitter to simulate inter-operator manual contouring variance.

3. **`03_radiomic_extraction.py`**:
   - IBSI-compliant 3D radiomic feature extraction using PyRadiomics across baseline ground truth, synthetic perturbations, and automated deep-learning segmentations.
   - Fixed bin width intensity discretization ($W = 25$) to maintain physical contrast interpretation.

4. **`04_icc_reproducibility_audit.py`**:
   - Quantifies two-way random single-rater absolute agreement intraclass correlation coefficients ($ICC[2,1]$), Lin's concordance correlation coefficient ($CCC$), and dynamic coefficient of variation ($CV$).
   - Stratifies features into Tier 1 (Robust: $ICC \ge 0.85, CV \le 30\%$), Tier 2 (Moderate: $0.70 \le ICC < 0.85$), and Tier 3 (Fragile: $ICC < 0.70$).

5. **`05_clinical_stress_testing.py`**:
   - Evaluates downstream predictive models under boundary noise via nested 5-fold cross-validation.
   - Compares the performance and calibration drift (AUC, Brier score loss) of unpruned naive pipelines versus stability-gated robust pipelines.

6. **`06_generate_figures_and_tables.py`**:
   - Generates publication-ready LaTeX tables (`table1_robust_atlas.tex`, `table2_stress_test.tex`, `table3_ablation.tex`) and visual artifacts from audit metrics.

7. **`run_pipeline.py`**:
   - Unified master runner providing fine-grained stage-by-stage or end-to-end execution.

## Dependencies and Installation

Install requirements using Python 3.9+:

```bash
pip install -r requirements.txt
```

Core dependencies:
- `numpy`, `scipy`, `pandas`, `scikit-learn`
- `SimpleITK`, `pyradiomics`
- `pingouin` (for ANOVA-based ICC calculations)
- `matplotlib`, `seaborn`
- `pyyaml`, `tqdm`

## Execution Guide

### End-to-End Execution
To run the full workflow sequentially:
```bash
python run_pipeline.py --step all
```

### Stage-by-Stage Execution
```bash
# 1. Dataset verification and clinical harmonization
python run_pipeline.py --step audit_data

# 2. Generate morphological and elastic perturbations
python run_pipeline.py --step perturb

# 3. Batch radiomic feature extraction
python run_pipeline.py --step extract

# 4. Statistical reproducibility auditing and tier classification
python run_pipeline.py --step icc_audit

# 5. Diagnostic stress testing and calibration evaluation
python run_pipeline.py --step stress_test

# 6. Export formatted tables and figures
python run_pipeline.py --step export_tables
```

## Configuration

All paths, voxel resampling dimensions, bin widths, structuring element radii, and stability thresholds are defined in `config.yaml`.
