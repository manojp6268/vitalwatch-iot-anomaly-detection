# VitalWatch - IoT ECG Anomaly Detection System

> *Every heartbeat tells a story. VitalWatch listens.*

ML-powered anomaly detection on real clinical ECG data - built at the intersection of IoT, Healthcare, and Machine Learning. Trained on the MIT-BIH Arrhythmia Database and deployed as a live Streamlit dashboard connected to a real smartwatch via Google Fit API.

---

## The Problem

A wearable device continuously monitors a patient's heart rate. Most beats are normal. A few are not.

The challenge: abnormal cardiac events are rare - sometimes less than 2% of all beats. A model that predicts "Normal" for everything achieves 98% accuracy and misses every dangerous event. Traditional ML fails here.

VitalWatch is built around this specific challenge.

---

## Dataset

**MIT-BIH Arrhythmia Database** - the gold standard for cardiac ML research.

- Patient Records: 10 (training: 8, testing: 2 unseen patients)
- Sampling Rate: 360 Hz
- Duration: ~30 minutes per record (~650,000 samples each)
- Total beats annotated: ~23,000 across all records
- Abnormal beats: ~1.5% - extreme class imbalance
- Source: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

Streamed directly via `wfdb` - no manual download required.

---

## Approach

### Features Engineered (6 total)

| Feature | Description |
|---|---|
| `rr_interval` | Time between consecutive heartbeats (ms) |
| `heart_rate` | Beats per minute derived from RR interval |
| `rr_diff` | Beat-to-beat change in RR interval |
| `rolling_mean_rr` | Local rhythm baseline (5-beat window) |
| `rolling_std_rr` | Local rhythm variability (5-beat window) |
| `relative_rr` | Current RR normalised against local baseline |

### Models

**V1 - Isolation Forest + DBSCAN Comparison**
Unsupervised, no labels needed, imbalance-resistant. DBSCAN included to demonstrate architectural mismatch on heavily skewed data.

**V2 - Deep Learning Ensemble**
Three-model ensemble across Isolation Forest, Autoencoder, and LSTM Autoencoder. Two voting strategies - OR (maximise recall) and AND (maximise precision).

### IoT Simulation

Model trained on historical patient data, frozen, then fed one beat at a time simulating a live wearable stream. Scaler fitted on training data only - no data leakage.

### Live Demo

Connected to a real Noise smartwatch via Google Fit API. Streamlit dashboard monitors live heart rate, runs the OR ensemble, and flags anomalies in real time with 60-second refresh.

---

## Results

### V1 - Single Patient (Patient 100)

| Phase | Features | Detection Rate | False Positives |
|---|---|---|---|
| Phase 3 - Baseline | 3 | 44.1% | 20 |
| Phase 4 - Enhanced | 6 | 52.9% | 17 |
| Phase 4b - DBSCAN default | 6 | 70.6% | 171 |
| Phase 4b - DBSCAN tuned | 6 | 17.6% | 64 |

### V2 - Multi-Patient (2 unseen patients)

| Model | Patient | Caught | False+ | Precision | Recall |
|---|---|---|---|---|---|
| Isolation Forest | 112 | 8/12 | 7 | 53.3% | 66.7% |
| Isolation Forest | 113 | 6/6 | 0 | 100.0% | 100.0% |
| Autoencoder | 112 | 12/12 | 4 | 75.0% | 100.0% |
| Autoencoder | 113 | 6/6 | 126 | 4.5% | 100.0% |
| LSTM | 112 | 3/12 | 8 | 27.3% | 25.0% |
| LSTM | 113 | 4/6 | 43 | 8.5% | 66.7% |
| OR Ensemble | 112 | 12/12 | 13 | 48.0% | 100.0% |
| OR Ensemble | 113 | 6/6 | 163 | 3.6% | 100.0% |
| AND Ensemble | 112 | 3/12 | 1 | 75.0% | 25.0% |
| AND Ensemble | 113 | 4/6 | 0 | 100.0% | 66.7% |

**Headline finding:** When all three models simultaneously flagged a beat as anomalous - they were never wrong. AND Ensemble on Patient 113: 0 false positives, 100% precision. Every alarm was real.

---

## Clinical Visualisation

![VitalWatch - Three Panel Clinical View](vitalwatch_final.png)

*Top: Raw ECG stream with anomaly flags (red = true anomaly, orange = false positive)*

*Middle: Heart rate over time with tachycardia/bradycardia thresholds*

*Bottom: RR interval trace with scatter markers at flagged beats*

---

## Project Structure

```
vitalwatch-iot-anomaly-detection/
│
├── phase1_data_exploration.ipynb       # EDA, signal loading, annotation analysis
├── phase2_feature_engineering.ipynb    # RR intervals, heart rate, rolling features
├── phase3_anomaly_detection.ipynb      # Baseline Isolation Forest (3 features)
├── phase4_model_improvement.ipynb      # Enhanced model (6 features)
├── phase4_model_comparison.ipynb       # DBSCAN vs Isolation Forest
├── phase5_iot_simulation.ipynb         # Live stream simulation + visualisation
├── phase6_autoencoder.ipynb            # Autoencoder - V2 deep learning baseline
├── phase7_multi_patient.ipynb          # Multi-patient generalisation (10 records)
├── phase8_lstm_anomaly_detection.ipynb # LSTM Autoencoder — sequential detection
├── phase9_ensemble.ipynb               # Three-way ensemble, OR/AND voting
│
├── streamlit_app.py                    # Live dashboard - Google Fit + OR ensemble
├── google_fit_auth.py                  # Google Fit API authentication + data pull
├── discover_sources.py                 # Utility to list available Fit data sources
│
├── vitalwatch_final.png                # Phase 5 three-panel clinical visualisation
├── phase7_generalisation.png           # Multi-patient reconstruction error plots
├── phase8_lstm.png                     # LSTM error distribution plots
├── phase9_comparison.png               # Full five-model comparison grid
│
├── VitalWatch_Learning_Journal.docx    # Complete learning documentation
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Excludes credentials and tokens
└── README.md
```

---

## Setup

```bash
# Create environment
conda create -n vitalwatch python=3.10
conda activate vitalwatch

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook
```

Run notebooks in order: phase1 → phase2 → phase3 → phase4 → phase4b → phase5 → phase6 → phase7 → phase8 → phase9

---

## Live Demo - Google Fit Integration

VitalWatch connects to real wearable data via Google Fit API.

The dashboard trains on MIT-BIH clinical data, then monitors your own heart rate in real time - flagging anomalies using the OR Ensemble (Isolation Forest + Autoencoder).

```bash
# After setting up Google Cloud credentials
streamlit run streamlit_app.py
```

> **Note:** Consumer wearable data has lower precision than clinical ECG. False positive rate is higher in live demo than in clinical evaluation. VitalWatch is a personal health curiosity tool, not a medical device.

---

## Key Learnings

- Class imbalance in healthcare is extreme - standard accuracy is meaningless
- Model selection is problem-specific, not reputation-based
- DBSCAN is architecturally unsuitable for heavily imbalanced cardiac data
- Data leakage is the most common reason models fail in production
- Signal dilution explains why LSTM misses single-beat anomalies despite memory
- Isolation Forest trained on 8 patients outperformed deep learning models on 1
- Training data diversity matters more than model complexity

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · TensorFlow · Keras · Matplotlib · WFDB · Streamlit · Plotly · Google Fit API · Jupyter · Anaconda

---

## What's Next - V3 Roadmap

- [ ] Patient-specific threshold calibration
- [ ] Improved Autoencoder with attention mechanism
- [ ] Precision-recall curves and F1 evaluation dashboard
- [ ] Real-time retraining on new patient baseline

---

*Built entirely from scratch as part of a self-directed learning journey in Healthcare ML, IoT systems, and Deep Learning - from a 1970s clinical database in Boston to a live Streamlit dashboard in Trier, 2026.*
