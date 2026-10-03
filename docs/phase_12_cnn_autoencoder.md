# Phase 12: CNN Autoencoder

This phase explores a newer deep learning direction based on 1D convolutional patterns.

## Why CNN here
The earlier feedforward Autoencoder treated each feature as a flat vector. A 1D CNN instead learns local structure across the feature sequence.

## What this may improve
A convolutional model may better capture patterns such as short RR interval + sharp RR difference + abnormal relative RR, rather than treating those values as unrelated.

## Current status
This phase is still under progress and should be treated as exploratory work rather than final evidence.

## Main lesson
The architecture may be more expressive, but it must still be judged with the same questions as before: class imbalance, calibration, and patient-specific drift.
