# UCI HAR Dataset

## Purpose

This directory is the original UCI Human Activity Recognition Using Smartphones dataset package. It includes activity metadata, 561 engineered features, participant labels, and raw inertial signals.

## Dataset Structure

| Location | Purpose |
|---|---|
| `activity_labels.txt` | Numeric activity IDs and names |
| `features.txt` | Names for the 561 feature columns |
| `features_info.txt` | Definitions of feature domains and sensor variables |
| `train/` | Training feature, label, participant, and raw signal files |
| `test/` | Test feature, label, participant, and raw signal files |

## Window Format

Each row in `X_train.txt` or `X_test.txt` represents a 128-sample, 2.56-second window. Raw signals contain nine channels:

- Total acceleration: X, Y, Z
- Body acceleration: X, Y, Z
- Body gyroscope: X, Y, Z

The repository's workflow pools the supplied partitions and creates a participant-held-out 80%/20% split for the experimental evaluation.

## Citation and Acknowledgement

The dataset should be cited according to the UCI dataset publication and the licensing terms referenced in the original `README.txt`. Use of the dataset in publications should acknowledge Reyes-Ortiz et al. and the UCI Human Activity Recognition project.
