# Training Data

## Purpose

This directory contains the original UCI HAR training partition, including engineered features, activity labels, participant IDs, and raw inertial signal windows.

## Files

- `X_train.txt` → 561 engineered features per window.
- `y_train.txt` → Numeric activity labels.
- `subject_train.txt` → Participant identifier for each window.
- `Inertial Signals/` → Nine raw channels for each 128-sample window.

## Use Case

The project loads this partition, combines it with the test partition, and performs a new participant-aware 80%/20% split. This prevents windows from the same participant appearing in both the development and final test sets.
