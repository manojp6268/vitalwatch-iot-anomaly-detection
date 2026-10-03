# The VitalWatch Learning Journey

This document captures the overall story of the VitalWatch project as a learning journey.

## The core question

The project began with a simple but challenging question: **Can a machine learning system detect abnormal heartbeats without ever seeing one during training?**

This constraint is not academic—it is the reality of deploying anomaly detection on wearable devices. The data streaming from a smartwatch is raw and unlabelled. The system must learn what "normal" looks like and detect deviations, without access to annotated examples of what "abnormal" means.

## Why this matters

In healthcare, a model that predicts "normal" for everything can still achieve 98%+ accuracy by raw numbers alone. But that is exactly the problem: the rare abnormal beats are the ones that matter most.

This project explores how to work under that constraint while still learning something meaningful from data that are overwhelmingly normal.

## The journey in phases

### Phase 1-2: Understanding the data
The project started by understanding the structure of ECG data: why raw signals are not directly useful for machine learning, and why feature engineering matters.

**Key learning**: Heart rhythm is often better summarized by beat-to-beat timing features (RR intervals) than by raw waveform samples.

### Phase 3: First anomaly detection
Isolation Forest was chosen as the first model because it works in an unsupervised setting and makes intuitive sense: anomalies are easier to isolate than normal points.

**Key learning**: A simple unsupervised model can work, but it depends heavily on the contamination prior—our assumption about how many anomalies exist.

### Phase 4: Comparison and critique
Different models were compared. DBSCAN exposed a different problem: it is highly sensitive to hyperparameters and fails when data distributions are extremely imbalanced.

**Key learning**: Not all algorithms are equally suited to all problems. A powerful algorithm can be the wrong tool for a specific setting.

### Phase 5: The real world
The project shifted to how the system would operate in a streaming or wearable-style environment. How would alerts be triggered? How would data flow? What latency constraints matter?

**Key learning**: The notebook experiment is one thing; the real deployment pipeline is another.

### Phase 6: Reconstruction learning
An Autoencoder was explored because it learns what a normal beat looks like and flags anything that cannot be reconstructed well.

**Key learning**: Unsupervised models can learn patterns by looking at reconstruction error, not just statistical outliers.

### Phase 7: The distribution shift problem
When the analysis moved beyond a single patient, a major realization hit: the distribution of abnormal beats changes dramatically from patient to patient.

**Key learning**: Patients are not interchangeable. One global threshold or prior does not work across all hearts.

### Phase 8: Sequence models and signal dilution
An LSTM-based Autoencoder was explored to capture temporal context. This led to a critical discovery: when errors are averaged across a 10-beat sequence, a single anomalous beat's signal can get diluted and lost.

**Key learning**: Architectural choices have unintended consequences. Aggregation functions matter deeply.

### Phase 9: Ensemble approaches
Different voting strategies were tested: OR ensemble (maximize recall) vs AND ensemble (maximize precision).

**Key learning**: There is no single "right" model—the choice depends on the clinical context and what trade-off matters more.

### Phase 11: Scale and benchmarking
The analysis was expanded to 46 MIT-BIH records and 108,098 beats. A classical baseline (Pan-Tompkins from 1985) was compared against.

**Key learning**: Learned unsupervised models can be competitive with fixed-rule approaches, even while remaining completely interpretable.

### Phase 12: Modern architectures
CNN-based autoencoders were explored to better capture local relationships between contiguous ECG-derived features.

**Key learning**: More complexity is not always better, but architectural choices should be intentional and evaluated carefully.

## The biggest insights

1. **Accuracy is not the metric that matters here.** In imbalanced healthcare data, accuracy lies to you. Precision, recall, and their tradeoffs are what matter.

2. **The imbalance problem is not just about ratio; it is about distribution shift.** Some patients are 99.9% normal. Others are 100% abnormal. One model cannot learn the same threshold for both.

3. **Patients are not interchangeable.** What constitutes "normal" for one person may be "abnormal" for another. Patient-specific calibration is not a luxury—it is necessary.

4. **A single contamination prior is risky.** Training the model assuming 1.5% anomalies and then deploying it on a patient with 30% anomalies produces systematic under-flagging.

5. **Explaining the model is just as important as getting a high score.** A clinician cannot act on "the model says anomaly" without understanding why. Explainability is not optional in healthcare.

6. **Model performance depends on training data diversity.** Isolation Forest trained on 8 patients outperformed LSTM trained on 1 patient. Data diversity > model complexity.

7. **Architectural choices have real consequences.** LSTM's mean error aggregation hides anomalies. DBSCAN's epsilon sensitivity causes collapse. These are not bugs to fix; they are fundamental architectural properties.

8. **The real value is understanding failure modes.** Every model that failed taught something crucial. The project is better for knowing why things don't work.

## Final reflection

The project started as "Can we build an anomaly detector?" and evolved into "What are the real constraints and tradeoffs in this specific problem?"

That shift from building to understanding is what makes this a learning journey rather than just an engineering project.

The final model is not perfect. But the understanding of why it succeeds and fails in specific ways is the real outcome.
