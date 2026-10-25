# Research Brief: PitNET MRI Radiomics Reproducibility & Clinical Phenotyping

## 1. Executive Summary

- **Title**: Robustness and Reproducibility of MRI Radiomic Features in Pituitary Neuroendocrine Tumors: Impact of Multi-Model Segmentation Uncertainty on Endocrine Subtype and Proliferation Profiling
- **Clinical Motivation**: Pituitary neuroendocrine tumors (PitNETs) require accurate pre-operative characterization of functional subtype (NFPA vs. hormone-secreting), invasiveness (Knosp grade / carotid encroachment), and proliferative activity (Ki-67). While radiomics studies claim high predictive AUCs, they rarely report feature robustness against inter-observer contour variations or automated segmentation shifts. This work systematically interrogates the stability of PitNET radiomics across synthetic and deep-learning segmentation perturbations, maps a certified "Robust Radiomics Atlas," and demonstrates the dramatic clinical penalty of deploying unvetted, fragile features.

## 2. Core Research Questions

1. **Feature Vulnerability**: How many conventional and high-order MRI radiomic features remain stable (ICC > 0.85, CCC > 0.85) when subjected to subtle boundary shifts (±1–2 mm erosion/dilation and boundary jitter) in PitNETs?
2. **Algorithmic Discrepancy**: When modern deep learning architectures (e.g., nnU-Net, CNN ensemble) segment PitNETs compared to expert human delineations, which feature families (shape, first-order, GLCM, GLRLM, GLSZM, wavelet) exhibit severe drift?
3. **Sequence Dependency**: How does feature stability contrast between contrast-enhanced T1-weighted (CE-T1) and T2-weighted MRI?
4. **Clinical Downstream Consequence (The Killer Experiment)**: When predicting endocrine subtypes (NFPA vs. hormone-active) and high proliferation (Ki-67 ≥ 3%), how severely do models trained on unpruned/fragile features degrade under boundary uncertainty compared to models restricted to validated robust features?

## 3. Cohort Overview

| Cohort | Source | Patients | Sequences | Segmentation Ground Truth | Automated Model Segmentations | Clinical / Endocrine Endpoints |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cohort 1 (Discovery & Endocrine Profiling)** | UCL Tertiary Neurosciences Center (Pandit et al., *Sci Data* 2025) | 136 | Isovolumetric CE-T1 (100%), co-registered T2 (subset) | Expert consensus (Tumor, Carotids) | Synthetic perturbations (erosion, dilation, elastic) | NFPA (n=85+), Cushing, Acromegaly, Prolactinoma, Ki-67 index, Knosp invasion |
| **Cohort 2 (Algorithmic & Multi-Sequence Audit)** | Prague Military University Hospital (Černý et al., *OpenNeuro ds006248*) | 50 | 3T multiparametric: Coronal CE-T1, T1, Sagittal CE-T1, Coronal T2, FIESTA, ADC | Multi-class expert consensus (Tumor, Gland, Carotids, Chiasm) | 5 models: nnU-Net (full), nnU-Net (replicated), CNN ensemble, 2.5D CNN, GrowCut | Boundary variation across AI models, cross-sequence transfer |

## 4. Methodological Deliverables

1. **PitNET Robust Radiomics Atlas**: Categorization of all 107+ IBSI-standard features into:
   - *Tier 1 (Robust)*: ICC ≥ 0.85 across both synthetic shifts and deep-learning segmentation models.
   - *Tier 2 (Moderate)*: 0.70 ≤ ICC < 0.85.
   - *Tier 3 (Fragile / High-risk)*: ICC < 0.70 (pruned from clinical deployment).
2. **Stress-Testing curves**: AUC, Brier score, and calibration drift curves of clinical classification models under increasing boundary noise levels (0 mm to 3 mm jitter).
3. **Open-Science Codebase**: Fully reproducible pipeline with configuration files, feature extraction configs, and statistical scripts.
