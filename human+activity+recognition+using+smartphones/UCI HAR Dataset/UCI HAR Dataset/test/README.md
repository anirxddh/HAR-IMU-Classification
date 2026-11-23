# Test Data

## Purpose

This directory contains the original UCI HAR test partition, including engineered features, activity labels, participant IDs, and raw inertial signal windows.

## Files

- `X_test.txt` → 561 engineered features per window.
- `y_test.txt` → Numeric activity labels.
- `subject_test.txt` → Participant identifier for each window.
- `Inertial Signals/` → Nine raw channels for each 128-sample window.

## Use Case

The project pools this data with the original training partition before creating the participant-aware 80%/20% evaluation split. The final test set is withheld until all model selection and deployment decisions are complete.
