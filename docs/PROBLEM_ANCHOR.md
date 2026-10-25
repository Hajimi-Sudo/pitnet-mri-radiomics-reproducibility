# Problem Anchor: The Reproducibility Bottleneck in Pituitary Radiomics

## 1. Clinical & Anatomical Context

Pituitary neuroendocrine tumors (PitNETs) arise within the confined space of the sella turcica, directly abutting critical neurovascular structures:
- Bilateral internal carotid arteries (ICA) within the cavernous sinuses
- The compressed residual normal anterior pituitary gland and posterior bright spot
- The optic chiasm and infundibulum superiorly
- The sphenoid sinus bone and mucosal lining inferiorly

In high-resolution contrast-enhanced MRI:
- Carotid flow voids or high enhancement inside the cavernous plexus create extreme local signal gradients.
- Normal pituitary tissue enhances avidly and earlier than most adenomas.
- Suprasellar cistern CSF provides dark contrast on T1 and bright contrast on T2.

## 2. The Core Problem: Boundary Contamination and "Spurious Biomarkers"

In existing literature, over 50 retrospective studies report machine learning models utilizing hundreds to thousands of filtered radiomic features to predict:
- Invasive growth into the cavernous sinus (Knosp grade 3-4)
- Pituitary tumor consistency (soft vs. fibrous, affecting suction resectability)
- Functional hormone hypersecretion (GH, ACTH, PRL)
- Proliferative aggressiveness (Ki-67 labeling index ≥ 3%)

**The Unaddressed Crisis**:
Due to the small lesion volume (often 1–5 cm³) and tortuous boundaries, a mere **1-to-2 voxel divergence** in manual contouring or automated deep-learning mask generation can inadvertently incorporate:
1. Slices of the highly enhancing cavernous sinus wall,
2. Incomplete exclusions of normal pituitary rim, or
3. Adjacent dura/sphenoid bone partial volume artifacts.

When higher-order texture filters (e.g., Wavelet high-pass, Co-occurrence Matrix entropy, Large Zone Emphasis) process these contaminated edges, the resulting feature values fluctuate by over 300%. Consequently:
- High predictive performance in single-observer cohorts is frequently an artifact of non-physiological edge leakage.
- When transferred to external centers or automated nnU-Net segmentations, models suffer catastrophic generalizability failure.

## 3. Literature Baselines and Benchmarks

- **Zheng et al., 2024 (*Front Endocrinol*, PMID: 39371931)**: Systematic review concluding that reproducibility of imaging features and lack of standardized workflows represent the greatest barriers preventing pituitary radiomics from clinical adoption.
- **Pandit et al., 2025 (*Sci Data*, PMID: 39814754)**: Release of the 136-patient UCL PitNET repository emphasizing the urgent need for external benchmarking and reproducible computational radiomics.
- **Černý et al., 2025 (*OpenNeuro ds006248*)**: Benchmark of 5 automated segmentation models demonstrating that even state-of-the-art nnU-Net models exhibit inter-algorithm Dice variations of 0.73–0.86, introducing unavoidable boundary variance in automated pipelines.
- **Image Biomarker Standardisation Initiative (IBSI)**: Recommends explicit reporting of test-retest and inter-segmentation ICC before feature selection, a criterion violated by >90% of published pituitary studies.

## 4. Innovation & Scientific Contribution of This Study

1. **First Comprehensive Perturbation Audit in PitNETs**: Quantifying both synthetic geometric shifts (erosion/dilation/elastic deformation) and actual deep-learning segmentation model discrepancies (5 models).
2. **Identification of the "PitNET Robust Radiomics Atlas"**: Establishing a public reference list of IBSI features that neuroendocrine researchers can safely rely on.
3. **The Clinical Stress-Test Experiment**: Quantifying the exact penalty of feature fragility on endocrine subtype and Ki-67 classification, proving that robustness-guided feature pre-filtering prevents model collapse in clinical validation.
