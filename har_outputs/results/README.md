# Experimental Results

## Purpose

This directory contains machine-readable CSV exports for each stage of model evaluation. Values should be treated as generated evidence rather than modified by hand.

## Files

| File | Contents |
|---|---|
| `4_tuning.csv` | Baseline and tuned grouped-CV macro-F1 plus selected settings |
| `5_cross_validation.csv` | Mean and standard deviation by model |
| `6_cv_metrics.csv` | Out-of-fold grouped-CV metrics and AUC |
| `7_deployment.csv` | Serialized model size and measured inference latency |
| `9_test_results.csv` | Final metrics on the unseen participant test set |

## Use Case

Open these tables in pandas or a spreadsheet tool for reporting and further analysis. The final test table includes the cross-validation-to-test F1 gap, which indicates generalization consistency.
