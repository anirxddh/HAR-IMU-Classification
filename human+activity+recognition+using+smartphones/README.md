# UCI HAR Dataset

## Purpose

This folder is a Git-visible guide for obtaining and arranging the UCI Human Activity Recognition Using Smartphones dataset. The dataset's large text and archive files are intentionally excluded from Git; this structure explains where to place the extracted files and how the repository uses them.

The guide follows the following workflow:

`Download/extract the UCI HAR dataset → preserve this folder structure → run the Python or notebook workflow`

## Contents

| Item | Purpose |
|---|---|
| `UCI HAR Dataset/` | Extracted dataset directory |
| `UCI HAR Dataset/README.md` | Archive-layout and extraction guide |
| `UCI HAR Dataset/UCI HAR Dataset/README.md` | Dataset-package guide |
| `UCI HAR Dataset/UCI HAR Dataset/train/` | Place the original training partition here |
| `UCI HAR Dataset/UCI HAR Dataset/test/` | Place the original test partition here |

## Extracting the Dataset

1. Download the UCI HAR dataset archive.
2. Extract it into this folder, without renaming the original nested `UCI HAR Dataset` directory.
3. Preserve the following layout:

```text
human+activity+recognition+using+smartphones/
└── UCI HAR Dataset/
    └── UCI HAR Dataset/
        ├── activity_labels.txt
        ├── features.txt
        ├── features_info.txt
        ├── README.txt
        ├── train/
        │   ├── X_train.txt
        │   ├── y_train.txt
        │   ├── subject_train.txt
        │   └── Inertial Signals/
        └── test/
            ├── X_test.txt
            ├── y_test.txt
            ├── subject_test.txt
            └── Inertial Signals/
```

The archive and large raw text files are ignored by Git. Only README guides and other small, useful metadata should be committed.

## Use Case

The source workflow resolves feature tables, activity labels, participant identifiers, and raw inertial signal files from this directory. The data is not generated or modified here.
