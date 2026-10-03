# Phase 12: CNN Autoencoder

This phase explores a newer deep learning direction based on 1D convolutional patterns.

## Why CNN here

The earlier feedforward Autoencoder treated each feature as a flat vector. A 1D CNN instead learns local structure across the feature sequence.

For example, instead of treating RR interval, RR difference, and relative RR as three independent numbers, a CNN learns that:
- "short RR interval + sharp RR difference + low relative RR" is a specific morphological signature
- These features have local relationships worth capturing

## What this may improve

A convolutional model may better capture patterns that a feedforward network misses because it explicitly models spatial/sequential relationships between adjacent features.

## Current status

This phase is still under progress and should be treated as exploratory work rather than final evidence.

## Main lesson

The architecture may be more expressive, but it must still be judged with the same questions as before:
- Does it handle class imbalance?
- How does it perform across different patient distributions?
- Does calibration mismatch still matter?
- Can it be explained?

## Files

- **Notebooks**:
  - `phase12_calibration_analysis.ipynb`
  - `phase12_cnn_autoencoder.ipynb`
