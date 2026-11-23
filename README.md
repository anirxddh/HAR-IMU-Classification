# Human Activity Recognition from Smartphone IMU Signals

## Introduction

This repository presents an end-to-end machine learning workflow for Human Activity Recognition (HAR) using triaxial accelerometer and gyroscope signals from smartphones. It compares four classical classifiers trained on 561 engineered features with a one-dimensional convolutional neural network (1D-CNN) trained on raw 128 × 9 sensor windows.

The project was developed to classify six everyday activities: walking, walking upstairs, walking downstairs, sitting, standing, and lying down. The implementation, experimental outputs, trained models, plots, and dataset are included for transparent evaluation and embedded-model analysis.

The workflow follows:

`Data collection → person-aware split → preprocessing → classical/CNN training → tuning → cross-validation → deployment and sizing → unseen-person testing`

## Problem Statement

Human activity recognition is useful for fitness monitoring, health assessment, assisted living, context-aware applications, and wearable systems. Accelerometer and gyroscope signals contain overlapping patterns, especially between sitting and standing or between different forms of walking. A reliable classifier must therefore generalize to people who were not present during training.

The central technical challenge is to:

1. Prevent data leakage caused by overlapping windows or repeated samples from the same person.
2. Compare traditional and deep-learning approaches consistently.
3. Evaluate generalization using completely unseen volunteers.
4. Measure model size and inference latency for deployment on mobile or embedded devices.

## Aim and Objectives

The objective is to build and evaluate a reproducible HAR system using the UCI Human Activity Recognition Using Smartphones dataset.

- Load 561 extracted features and nine raw inertial signal channels.
- Split the data by participant rather than by random window.
- Preprocess the training data and prevent test-set leakage.
- Tune K-Nearest Neighbours, SVM, Random Forest, and Gaussian Naive Bayes classifiers.
- Train a compact 1D-CNN on raw sensor windows.
- Perform grouped five-fold cross-validation.
- Export classical models and TensorFlow Lite CNN variants.
- Compare accuracy, precision, recall, macro-F1, AUC, size, and latency.

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Aim and Objectives](#aim-and-objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Workflow](#running-the-workflow)
- [Artifacts and Output](#artifacts-and-output)
- [Model Selection](#model-selection)
- [Limitations](#limitations)
- [License](#license)

## Dataset

The repository includes the UCI HAR dataset collected from 30 volunteers aged 19–48 years. Participants wore a Samsung Galaxy S II at the waist and performed six activities while accelerometer and gyroscope signals were recorded at 50 Hz.

| Activity | Label |
|---|---:|
| Walking | 1 |
| Walking upstairs | 2 |
| Walking downstairs | 3 |
| Sitting | 4 |
| Standing | 5 |
| Lying down | 6 |

Sensor readings were divided into fixed windows of 128 samples, corresponding to 2.56 seconds with 50% overlap. The dataset provides:

- 561 named, time-domain and frequency-domain features.
- 128-sample windows for total acceleration, body acceleration, and body gyroscope, split across X/Y/Z axes.
- Participant identifiers.
- Activity labels.

This implementation combines the supplied train and test partitions and creates a new **80%/20% split by volunteer**: 24 participants for training and 6 for final testing. This differs from the original 70%/30% participant partition and ensures that no participant's windows appear in both sets.

## Methodology

### 1. Data loading and participant-aware splitting

The scripts resolve the local UCI HAR dataset, load feature names, class labels, participant IDs, and nine raw signal channels, then create a grouped split using `GroupShuffleSplit`. All later preprocessing and tuning statistics are derived only from training participants.

### 2. Preprocessing

- Missing values are counted before model training.
- Z-scores are used to examine outliers.
- Values beyond ±3 standard deviations are reported but retained because sensor outliers can represent meaningful movement.
- `StandardScaler` is used in pipelines for distance-based and scale-sensitive models.
- Principal component analysis is tested as an optional feature-reduction step.
- Class balance is measured; the reported evaluation uses macro-averaged metrics.
- Raw CNN channels are normalized using means and standard deviations calculated from training participants only.

### 3. Classical models

| Model | Learning approach |
|---|---|
| K-Nearest Neighbours | Compares a window with nearby stored training samples. |
| Support Vector Machine | Learns boundaries between activity classes, with RBF and linear kernels evaluated. |
| Random Forest | Aggregates a tuned ensemble of decision trees. |
| Gaussian Naive Bayes | Uses class-conditional Gaussian feature distributions. |

Classical models operate through pipelines containing scaling, optional PCA, and the classifier. Hyperparameters are selected using grouped five-fold cross-validation, which keeps participants separate between folds.

### 4. 1D-CNN

The CNN receives shape `(128, 9)` raw-window tensors and uses convolution, pooling, global average pooling, dropout, and a six-class output layer. Training includes person-aware validation and early stopping to limit overfitting.

### 5. Evaluation

Primary ranking uses macro-F1 because all six activities should contribute equally. Accuracy, macro precision, macro recall, one-vs-rest AUC, confusion matrices, and ROC curves are also recorded. The untouched 20% participant test set is used only for final evaluation.

### 6. Deployment analysis

Classical models are serialized with Joblib. The trained CNN is exported as:

- Native Keras `.keras` model.
- Float32 TensorFlow Lite model.
- Int8-quantized TensorFlow Lite model.

Deployment outputs include serialized size and single-window inference latency.

## Key Results

### Grouped cross-validation

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| KNN | 89.0% | 0.885 |
| SVM | 94.5% | 0.943 |
| Random Forest | 93.3% | 0.932 |
| Naive Bayes | 80.2% | 0.792 |
| CNN (validation) | — | 0.918 |

The tuned SVM is the strongest classical model in grouped cross-validation.

### Final unseen-participant test

| Model | Accuracy | Macro-F1 | AUC | Size |
|---|---:|---:|---:|---:|
| SVM | 89.7% | 0.904 | 0.986 | 7.20 MB |
| Random Forest | 87.2% | 0.873 | 0.987 | 18.73 MB |
| KNN | 83.7% | 0.842 | 0.973 | 35.29 MB |
| Naive Bayes | 79.0% | 0.784 | 0.960 | 494 KB |
| CNN (Keras) | 81.4% | 0.816 | 0.956 | — |
| CNN TFLite int8 | 81.3% | 0.816 | 0.955 | 34 KB |

The SVM is the strongest classical test-set model: **89.7% accuracy** and **0.904 macro-F1**. It exceeds the project's 0.90 macro-F1 target on unseen participants.

For embedded deployment, the quantized CNN is the smallest artifact at approximately **34 KB** and has a measured inference latency of approximately **0.07 ms/window**. This compact size comes with lower test accuracy than the SVM and should be assessed against the target platform's memory and accuracy constraints.

> Results are from the repository's person-held-out split and saved generated artifacts. Exact timing varies with hardware and runtime; model size values are relatively stable.

## Repository Structure

| Location | Purpose |
|---|---|
| [`har_ai_ml.py`](./har_ai_ml.py) | Main reproducible Python workflow. |
| [`HAR_AI_ML.ipynb`](./HAR_AI_ML.ipynb) | Notebook-form implementation and report narrative. |
| [`HAR_ML.pdf`](./HAR_ML.pdf) | Rendered project report or presentation document. |
| [`HAR_ML.docx`](./HAR_ML.docx) | Editable project report or manuscript. |
| [`har_outputs/`](./har_outputs/) | Generated figures, trained models, and tabular results. |
| [`human+activity+recognition+using+smartphones/`](./human+activity+recognition+using+smartphones/) | UCI HAR dataset and local data documentation. |
| [`LICENSE`](./LICENSE) | MIT license. |

## Getting Started

### Prerequisites

- Python 3.9 or later.
- NumPy and pandas.
- Matplotlib and seaborn.
- scikit-learn.
- Joblib.
- TensorFlow/Keras for CNN training and TFLite conversion.

### Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn joblib tensorflow
```

Keep the UCI dataset in the repository's dataset location or update `DATA_DIR`/`WORK` in the script. The project expects the nested `UCI HAR Dataset/UCI HAR Dataset` directory structure.

## Running the Workflow

### Python script

Open [`har_ai_ml.py`](./har_ai_ml.py) and execute it in an environment containing the dependencies above. The script contains Google Colab mount and extraction cells, so local execution may require adapting those initial dataset-loading cells.

The script generates data to paths configured as:

- `WORK / "figures"`
- `WORK / "models"`
- `WORK / "results"`

In the current repository, the outputs are organized under [`har_outputs/`](./har_outputs/).

### Jupyter notebook

Open [`HAR_AI_ML.ipynb`](./HAR_AI_ML.ipynb) in Jupyter or Google Colab and run the cells sequentially. TensorFlow-dependent cells are skipped when TensorFlow is unavailable.

## Artifacts and Output

- [`har_outputs/figures/`](./har_outputs/figures/) — preprocessing plots, cross-validation plots, confusion matrices, ROC curves, and deployment trade-off charts.
- [`har_outputs/models/`](./har_outputs/models/) — serialized classical models and TensorFlow Lite/Keras artifacts.
- [`har_outputs/results/`](./har_outputs/results/) — tuning, validation, metrics, deployment, and unseen-test result tables.

Generated CSV files preserve the numerical evidence for reporting, and PNG files preserve the visual evidence.

## Model Selection

The SVM is recommended as the repository's primary high-accuracy classical deployment candidate. It achieves the highest test macro-F1 among the classical classifiers and remains compact compared with KNN and Random Forest.

The int8 TensorFlow Lite CNN is the preferred candidate for severe memory constraints, such as small microcontrollers, only if its lower accuracy is acceptable. Naive Bayes is useful as a tiny, fast baseline but does not clear the target macro-F1 threshold.

## Limitations

- Hyperparameters are selected with cross-validation, so cross-validation estimates can be slightly optimistic.
- The final test set is person-held-out but represents the same UCI HAR collection protocol and environment.
- Inference timing was measured on the development environment and is not a device benchmark.
- CNN performance could change with retraining hardware, TensorFlow versions, or random initialization.
- The analysis does not include real-time streaming, sensor drift, variable window alignment, or deployment to a physical microcontroller.

## License

This repository is distributed under the MIT License. The included UCI HAR dataset is subject to the dataset's original licensing and publication terms.
