# Study Design: Multi-Cohort Perturbation Audit & Clinical Stress Testing

## 1. Experimental Workflow Architecture

```
                                  [ Cohort 1: Pandit et al. (n=136) ]        [ Cohort 2: Černý et al. (n=50) ]
                                          High-Res CE-T1 + T2                      Multiparametric 3T MRI
                                                    │                                          │
                                                    ▼                                          ▼
                                     [ Preprocessing & Standardization ]        [ Preprocessing & Standardization ]
                                     • N4ITK Bias Field Correction              • Multi-sequence Co-registration
                                     • 1.0mm isotropic resampling               • 1.0mm isotropic resampling
                                     • Z-score intensity normalization          • Multi-class label isolation
                                                    │                                          │
                   ┌────────────────────────────────┴──────────────────┐                       │
                   ▼                                                   ▼                       │
       [ Synthetic Perturbation Engine ]              [ Real-World Algorithmic Perturbations ]◄┘
       • Dilation: +1, +2 voxels                      • Expert Ground Truth (GT)
       • Erosion: -1, -2 voxels                       • nnU-Net (Černý et al.)
       • Elastic B-spline jitter                      • nnU-Net (Da Mutten et al.)
       • Rigid spatial shift (±1 mm)                  • CNN Ensemble & GrowCut
                   │                                                   │
                   └──────────────────────────┬────────────────────────┘
                                              ▼
                             [ IBSI-Standard Radiomics Extraction ]
                             (PyRadiomics: Shape, 1st Order, GLCM, GLRLM, GLSZM, GLDM)
                             • Fixed Bin Width (FBW=25) vs. Fixed Bin Number (FBN=32)
                             • Native space vs. Wavelet decomposed sub-bands
                                              │
                                              ▼
                             [ Statistical Reproducibility Audit ]
                             • Two-way random absolute agreement ICC(2,1)
                             • Lin's Concordance Correlation Coefficient (CCC)
                             • Dynamic Coefficient of Variation (CV)
                             • Definition of the "PitNET Robust Radiomics Atlas"
                                              │
                                              ▼
                             [ Clinical Endocrine Downstream Stress-Test ]
                             • Task A: Functional (GH/ACTH/PRL) vs. NFPA
                             • Task B: Proliferative Aggressiveness (Ki-67 ≥ 3% vs. < 3%)
                             • Task C: Knosp invasion / Carotid encasement
                                              │
                             ┌────────────────┴────────────────┐
                             ▼                                 ▼
                     [ Pipeline A: Naive ]          [ Pipeline B: Robust ]
                     • All features                 • Filtered (ICC ≥ 0.85)
                     • LASSO + Classifier           • LASSO + Classifier
                             │                                 │
                             └────────────────┬────────────────┘
                                              ▼
                                 [ Boundary Attack / Perturbation ]
                                 • Inject 1-2 voxel edge jitter at inference
                                 • Deploy across 5 automated segmentation models
                                 • Plot Performance Degradation (ΔAUC, ΔBrier)
```

---

## 2. Quantitative Metrics & Decision Gates

### 2.1 Stability Classification Criteria

| Tier | Name | Criteria | Recommendation for Clinical Radiomics |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Robust** | $ICC(2,1) \ge 0.85$ and $CCC \ge 0.85$ across synthetic shifts AND inter-algorithm masks | Safe for inclusion in clinical biomarkers and diagnostic nomograms |
| **Tier 2** | **Moderately Stable** | $0.70 \le ICC(2,1) < 0.85$ | Usable only with ensemble contour averaging or automated uncertainty filtering |
| **Tier 3** | **Fragile / Volatile** | $ICC(2,1) < 0.70$ or $CV > 30\%$ | Must be pruned prior to machine learning; high risk of spurious overfit |

### 2.2 Mathematical Formulations

- **Intraclass Correlation Coefficient (ICC(2,1))**:
  $$ICC(2,1) = \frac{MS_R - MS_E}{MS_R + (k-1)MS_E + \frac{k}{n}(MS_C - MS_E)}$$
  where $MS_R$ is mean square between subjects, $MS_E$ is mean square error, $MS_C$ is mean square between perturbations/raters, $k$ is number of perturbations, and $n$ is patient count.
- **Lin's Concordance Correlation Coefficient (CCC)**:
  $$\rho_c = \frac{2 s_{xy}}{s_x^2 + s_y^2 + (\bar{x} - \bar{y})^2}$$
- **Stress-Test Robustness Ratio (RR)**:
  $$RR = \frac{AUC_{perturbed}}{AUC_{baseline}}$$
  A clinically robust pipeline must maintain $RR \ge 0.90$ under 1–2 voxel boundary shifts, whereas fragile pipelines typically drop below $RR < 0.75$.
