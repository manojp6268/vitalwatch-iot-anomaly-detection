# Phase 11: Baseline Comparison and Extended Evaluation

This phase scaled the project from single-patient to full-dataset analysis and introduced clinical baseline comparison.

## What changed

The analysis expanded from a single patient (Phase 3) to 46 MIT-BIH records containing 108,098 total beats.

This revealed a fundamentally different reality than the initial single-record assumptions.

## Key findings

### Class distribution heterogeneity
- Some records are almost entirely abnormal (100% abnormal beats, no normal beats)
- Some records have almost no abnormal beats (99.9% normal, only 2 abnormal beats)
- The overall database is 69.2% normal and 30.8% abnormal
- But individual patient distributions range from 0.08% to 100% abnormal

This is not just class imbalance—this is **distribution heterogeneity**.

### Performance variation
The same model achieves vastly different F1 scores depending on which record it is tested on:
- F1 = 0.7 on record 106 (high anomaly rate, extreme beats)
- F1 = 0.0 on record 100 (very rare anomalies, subtle patterns)

**This is not a model failure. This is data reality.**

## Baseline comparison: Pan-Tompkins

Pan-Tompkins is a classical signal processing algorithm published in 1985. It detects R-peaks and flags anything outside normal RR bounds as anomalous.

### Results
- Pan-Tompkins F1 = 0.525 (on 10 records)
- VitalWatch OR Ensemble F1 = 0.556
- VitalWatch AND Ensemble precision = 0.750 (when it flags, it is right)

VitalWatch outperforms the 40-year-old standard, using **zero labelled data**.

### But the nuance matters
Pan-Tompkins fails catastrophically on some records:
- Record 113: Flagged 1,626 beats as anomalous. Only 5 were real. Precision = 0.002.

VitalWatch learned patterns; Pan-Tompkins applied fixed rules. Rules work on extreme cases but fail on subtle ones.

## Why this matters

This phase revealed the core challenge: **the model is not wrong; the data distribution is complex.**

One contamination prior, one threshold, one global model—none of these are sufficient when patients are heterogeneous.

## Lesson for future work

Patient-specific calibration is not a luxury. It is necessary.

## Files

- **Notebooks**: 
  - `phase11a_full_dataset_loading.ipynb`
  - `phase11b_extended_evaluation.ipynb`
  - `phase11c_confusion_matrices.ipynb`
  - `phase11d_baseline_comparison.ipynb`
