# VitalWatch: IoT ECG Anomaly Detection

> Every heartbeat tells a story. ML-powered anomaly detection on simulated wearable vitals.

This repository is a learning journal and project archive documenting my journey building an ECG anomaly detection system. The work is intentionally exploratory, reflective, and phase-based rather than polished as a formal paper.

## Why this repo looks like a journal

This project was designed as a personal research log:
- each phase captures a specific question or experiment
- most notebooks are incremental builds rather than final production code
- findings are documented alongside code and results
- several sections intentionally preserve the learning process and mistakes

This is not a finished product in the strict engineering sense. It is a real research journey.

## Repository structure

```text
vitalwatch-iot-anomaly-detection/
├── README.md                              # This file
├── requirements.txt                       # Dependencies
├── streamlit_app.py                       # Live monitoring dashboard
├── google_fit_auth.py                     # Google Fit integration
│
├── docs/                                  # Learning journal (phase-by-phase)
│   ├── LEARNING_JOURNEY.md                # Full narrative overview
│   ├── phase_1_ecg_basics.md              # ECG understanding & data exploration
│   ├── phase_2_feature_engineering.md     # RR intervals, heart rate, features
│   ├── phase_3_isolation_forest.md        # First anomaly detection model
│   ├── phase_11_baseline_comparison.md    # Full dataset & Pan-Tompkins comparison
│   └── phase_12_cnn_autoencoder.md        # 1D CNN for spatial feature relationships
│
├── notebooks/                             # Jupyter notebooks (executable code)
│   ├── phase1_data_exploration.ipynb
│   ├── phase2_feature_engineering.ipynb
│   ├── phase3_anomaly_detection.ipynb
│   ├── phase4_model_comparison.ipynb
│   ├── phase4_model_improvement.ipynb
│   ├── phase5_iot_simulation.ipynb
│   ├── phase6_autoencoder.ipynb
│   ├── phase7_multi_patient.ipynb
│   ├── phase8_lstm_anomaly_detection.ipynb
│   ├── phase9_ensemble.ipynb
│   ├── phase11a_full_dataset_loading.ipynb
│   ├── phase11b_extended_evaluation.ipynb
│   ├── phase11c_confusion_matrices.ipynb
│   ├── phase11d_baseline_comparison.ipynb
│   ├── phase12_calibration_analysis.ipynb
│   └── phase12_cnn_autoencoder.ipynb
│
├── data/                                  # Datasets & processed results
│   ├── phase11_combined_dataset.parquet
│   ├── phase11_record_summary.csv
│   └── phase11_per_record_results.csv
│
├── results/                               # Visualizations & analysis outputs
│   ├── phase11_baseline_comparison.png
│   ├── phase11_confusion_matrices.png
│   ├── phase11_dataset_distribution.png
│   ├── phase11_evaluation_results.csv
│   ├── phase11_f1_heatmap.png
│   ├── phase11_final_summary.png
│   ├── phase11_pr_roc_curves.png
│   ├── phase7_generalisation.png
│   ├── phase8_lstm.png
│   ├── phase9_comparison.png
│   └── vitalwatch_final.png
│
└── VitalWatch.txt                         # Full learning narrative (text version)
```

## What is inside

### docs/
Written narrative: reasoning, assumptions, mistakes, discoveries, and reflections behind each phase.

### notebooks/
Code-heavy experimentation. Run notebooks in order for the full learning progression.

### data/
Processed datasets, summaries, and result tables.

### results/
Plots, confusion matrices, F1 heatmaps, and evaluation visuals.

## Project flow

The project roughly moves in this order:
1. Understand ECG basics and MIT-BIH data structure
2. Build feature engineering around RR intervals and heart rate
3. Try baseline unsupervised anomaly detection (Isolation Forest)
4. Compare different models (DBSCAN, Autoencoders, LSTMs)
5. Simulate live monitoring scenarios
6. Test ensemble voting logic and calibration issues
7. Scale to full-dataset evaluation and compare with clinical baselines
8. Explore CNN-based reconstruction methods

## Main themes

- **Class imbalance** in medical data
- **Unsupervised anomaly detection** without labels
- **Patient-specific calibration mismatch** across distributions
- **Precision vs recall tradeoff** in healthcare
- **Why one algorithm works for one setting but fails elsewhere**
- **Interpretability requirements** in healthcare
- **The gap between notebooks and real-world deployment**

## Key takeaways

Some of the big lessons from the project:
- Isolation Forest is simple but effective in unsupervised, imbalanced settings
- DBSCAN is hypersensitive to hyperparameters and data distribution
- LSTM sequence models dilute anomaly signals when errors are averaged across windows
- Autoencoders catch more anomalies but generate more false positives
- Ensemble strategies offer useful precision/recall tradeoffs
- A fixed contamination prior is risky when patient distributions vary widely
- Model performance depends heavily on training data diversity
- Clinical explainability matters as much as raw metrics

## Quick start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Read the learning journey
Start with `docs/LEARNING_JOURNEY.md` for the narrative overview, then explore individual phase notes.

### Run notebooks
```bash
jupyter notebook
```
Start with `phase1_data_exploration.ipynb` and progress through phases.

### Launch live dashboard
```bash
streamlit run streamlit_app.py
```

## Dataset

**MIT-BIH Arrhythmia Database** - 46 records, 108,098 beats, expert-annotated
- Sampling rate: 360 Hz
- Class distribution: 69.2% normal, 30.8% abnormal (with extreme per-record heterogeneity)
- Accessed via `wfdb` library; no manual download required

## Key findings

| Finding | Impact |
|---------|--------|
| **Class Imbalance ≠ Heterogeneity** | 1.5% abnormal on one patient vs 30.8% overall |
| **LSTM signal dilution** | Averaging reconstruction errors hides single anomalies |
| **AND Ensemble = zero false positives** | When all models agree, they're right |
| **Learned models beat Pan-Tompkins (1985)** | VitalWatch F1=0.549 vs baseline F1=0.524 |
| **Per-record performance varies widely** | Same model: F1=0.7 on one record, F1=0.0 on another |
| **Calibration prior mismatch** | Training on 1.5% assumption, testing on 30.8% reality |
| **Patient-specific calibration needed** | One threshold does not fit all hearts |

## Future work

- [ ] Fix LSTM signal dilution with maximum error aggregation
- [ ] Adaptive thresholding from incoming patient data
- [ ] OCSVM baseline comparison
- [ ] SHAP feature importance for explainability
- [ ] Leave-one-patient-out cross-validation
- [ ] Real smartwatch integration with higher-frequency ECG

## Notes

- **Learning journal style**: This is intentionally documentation-heavy and reflective, not a polished product README
- **Phase 12 (CNN)** is under progress
- **Not a medical device**: Proof-of-concept pipeline demonstration only
- **Google Fit integration** requires OAuth2 setup

## Status

- Learning journal: active
- Core experimentation: mostly complete (Phases 1-11)
- CNN extension: in progress (Phase 12)
- Deployment layer: proof of concept available

---

This repo is best approached as a research notebook archive and learning record rather than a clean production codebase. All phases are meant to be read as a progression: a journey, not a destination.
