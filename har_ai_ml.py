"""
# Human Activity Recognition (HAR) from Smartphone IMU Signals
**EC24313 · Modules IV & V · Problem #18**

## Setup — imports and settings

- `numpy` / `pandas`: arrays and tables · `matplotlib` / `seaborn`: plots
- `sklearn`: the four Module-IV models, the pipeline tools and the metrics (Module-V tool #1)
- `tensorflow` / `keras`: the neural network and TFLite conversion (Module-V tool #2)

`RANDOM_STATE = 42` fixes every random choice (the split, the forest, the network's starting weights).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import zipfile, urllib.request, time, os, warnings, joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report, ConfusionMatrixDisplay, roc_curve)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")

WORK = Path("/content") if Path("/content").exists() else Path.cwd()
FIG_DIR, MODEL_DIR, RES_DIR = WORK / "figures", WORK / "models", WORK / "results"
for d in (FIG_DIR, MODEL_DIR, RES_DIR):
    d.mkdir(exist_ok=True)

def savefig(name):
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
    print("TensorFlow", tf.__version__)
except ImportError:
    HAS_TF = False
    print("TensorFlow not found: the CNN and TFLite parts will be skipped")
print("Setup OK. Outputs go to:", WORK)

"""# Stage 1 — Data Collection

## 1.1 Download the dataset
**What the data is:** 30 volunteers wore a Samsung smartphone on their waist and did six activities (walking, walking upstairs, walking downstairs, sitting, standing, lying down). The phone's **accelerometer** (straight-line movement) and **gyroscope** (rotation) were recorded 50 times a second.

The signal was cut into **windows of 2.56 seconds (128 readings)**. For every window, the dataset authors also computed **561 summary features** (means, standard deviations, energies, FFT values and so on).

The zip contains a second zip inside it, so the code unzips twice. If the download fails, download the zip from the UCI page (dataset ID 240), drag it into Colab's file panel, rename it `har_outer.zip` and re-run.
"""

from google.colab import drive
drive.mount("/content/drive")

ZIP = "/content/drive/MyDrive/UCI_HAR_Dataset.zip"
!unzip -q -o "{ZIP}" -d /content/

DATA_DIR = Path("/content/UCI HAR Dataset")
print("Data folder found:", DATA_DIR.exists())

"""## 1.2 Load features, labels, volunteers and raw signals
- **Feature names:** `features.txt` names the 561 columns. A few names repeat (the `bandsEnergy` ones), so the code adds `__1`, `__2` to make them unique. Mention this in your report: it shows you inspected the data rather than loading it blindly.
- **How to read a name:** `t` = time domain, `f` = frequency domain (after FFT), `Body` = movement, `Gravity` = the slow gravity part, `Acc` = accelerometer, `Gyro` = gyroscope, `Jerk` = how fast acceleration changes, `Mag` = overall magnitude.
- **Raw signals:** the 9 channels (total acceleration, body acceleration and gyroscope, each on X/Y/Z) are stacked into shape `(windows, 128, 9)`. The four classical models use the 561 features; the CNN reads the raw signals directly.
"""

features = pd.read_csv(DATA_DIR / "features.txt", sep=r"\s+", header=None, names=["idx", "name"])
seen, FEATURE_NAMES = {}, []
for n in features["name"]:
    seen[n] = seen.get(n, -1) + 1
    FEATURE_NAMES.append(n if seen[n] == 0 else f"{n}__{seen[n]}")
print(f"{len(FEATURE_NAMES)} features; {sum(v > 0 for v in seen.values())} duplicated names fixed")

labels = pd.read_csv(DATA_DIR / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "activity"])
LABEL_MAP = dict(zip(labels["id"], labels["activity"]))
ACTIVITIES = list(labels["activity"])          # natural order, used for plots

SIGNALS = [f"{k}_{a}" for k in ["total_acc", "body_acc", "body_gyro"] for a in "xyz"]

def load_part(part):
    f = DATA_DIR / part
    X = pd.read_csv(f / f"X_{part}.txt", sep=r"\s+", header=None, names=FEATURE_NAMES)
    y = pd.read_csv(f / f"y_{part}.txt", header=None).squeeze("columns").map(LABEL_MAP)
    s = pd.read_csv(f / f"subject_{part}.txt", header=None).squeeze("columns")
    raw = np.stack([np.loadtxt(f / "Inertial Signals" / f"{sig}_{part}.txt") for sig in SIGNALS], axis=-1)
    return X, y, s, raw

parts = [load_part(p) for p in ("train", "test")]
X_all   = pd.concat([p[0] for p in parts], ignore_index=True)
y_all   = pd.concat([p[1] for p in parts], ignore_index=True)
subj_all = pd.concat([p[2] for p in parts], ignore_index=True)
raw_all = np.concatenate([p[3] for p in parts])

print("All windows:", X_all.shape, "| raw:", raw_all.shape, "| volunteers:", subj_all.nunique())

"""## 1.3 Split: 80% training / 20% test, **by person**
The dataset ships with a 70/30 split, but the workflow asks for **80/20**. So we pool all 30 volunteers and re-split.

**Why split by person and not by random rows?** Neighbouring windows overlap by 50%, and one person's walking looks a lot like their other walking windows. If the same person appears in both train and test, the model partly *memorises that person*, and test accuracy comes out higher than it would be for a stranger. A real wearable always meets a new person. So **24 volunteers train the model and 6 completely different volunteers test it**.

`GroupShuffleSplit` does this: it treats each volunteer as one block that can't be broken up.
"""

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X_all, y_all, groups=subj_all))

X_train, X_test = X_all.iloc[train_idx].reset_index(drop=True), X_all.iloc[test_idx].reset_index(drop=True)
y_train, y_test = y_all.iloc[train_idx].reset_index(drop=True), y_all.iloc[test_idx].reset_index(drop=True)
subj_train, subj_test = subj_all.iloc[train_idx].reset_index(drop=True), subj_all.iloc[test_idx].reset_index(drop=True)
raw_train, raw_test = raw_all[train_idx], raw_all[test_idx]

print(f"Train: {len(X_train)} windows from {subj_train.nunique()} people ({len(X_train)/len(X_all):.0%})")
print(f"Test : {len(X_test)} windows from {subj_test.nunique()} people ({len(X_test)/len(X_all):.0%})")
print("Test volunteers:", sorted(subj_test.unique()))
print("Overlap between train and test people:", set(subj_train) & set(subj_test) or "none ✅")

"""# Stage 2 — Data Preprocessing
All statistics here are computed on **training data only**.

## 2.1 Missing value handling
Models fail or behave badly when values are missing (NaN), so we count them.
"""

missing = X_train.isna().sum().sum()
print("Missing values in training features:", missing)
print("Decision:", "nothing to fill ✅" if missing == 0 else "fill with the training-set median")

"""## 2.2 Outlier detection
**Method:** the z-score. For each feature, we measure how many standard deviations each value is from that feature's mean. A value beyond **±3** is a statistical outlier.

**The decision matters more than the count.** In sensor data, an "outlier" is often a real event: a hard footstep on the stairs produces a big acceleration spike. Deleting those rows would delete exactly the information that separates walking downstairs from walking. The boxplot shows this: the "outliers" sit inside the moving activities.

**Decision:** report them and keep them. The features are already bounded to [-1, 1] by the dataset authors, so no single value can be wildly out of range.
"""

z = (X_train - X_train.mean()) / X_train.std()
frac_values = (z.abs() > 3).values.mean()
print(f"Values beyond ±3σ: {frac_values:.2%} of all feature values")
print(f"Value range: [{X_train.values.min():.2f}, {X_train.values.max():.2f}]")

feat = "tBodyAccMag-mean()" if "tBodyAccMag-mean()" in X_train.columns else X_train.columns[0]
plt.figure(figsize=(10, 4))
sns.boxplot(x=y_train, y=X_train[feat], order=ACTIVITIES)
plt.xticks(rotation=25); plt.xlabel(""); plt.title(f"Outliers live in the moving activities ({feat})")
savefig("2_2_outliers_boxplot"); plt.show()

"""## 2.3 Normalisation / scaling
**Why:** KNN and SVM measure *distances* between windows. If one feature ranged 0–1000 and another 0–1, the big one would dominate the distance. `StandardScaler` rescales every feature to **mean 0, standard deviation 1**, so all features count equally.

**Where it's applied:** Fitting a scaler on all data before splitting is the most common hidden leakage bug, and the pipeline prevents it.
"""

demo_cols = list(X_train.columns[:3])
scaled_demo = pd.DataFrame(StandardScaler().fit_transform(X_train[demo_cols]), columns=demo_cols)
pd.concat({"before": X_train[demo_cols].agg(["mean", "std"]),
           "after":  scaled_demo.agg(["mean", "std"])}, axis=1).round(3)

"""## 2.4 Oversampling (class imbalance check)
**Why:** if one activity had 10× more windows than another, a lazy model could guess the big class and still score well. Oversampling (e.g. SMOTE) creates extra samples of the small classes to fix that.

**Rule used:** imbalance ratio = biggest class ÷ smallest class. Above **1.5**, oversample (inside the CV folds, never before splitting). At or below 1.5, the classes are balanced enough, and we report **macro**-averaged metrics, which weight every class equally anyway.
"""

counts = y_train.value_counts().reindex(ACTIVITIES)
ratio = counts.max() / counts.min()
print(counts.to_string(), f"\n\nImbalance ratio: {ratio:.2f}")
print("Decision:", "oversample the minority classes" if ratio > 1.5
      else "balanced enough → no oversampling; use macro-averaged metrics ✅")

ax = (counts / counts.sum() * 100).plot(kind="bar", figsize=(9, 3.5), rot=25, color="#4C72B0")
ax.set_ylabel("% of training windows"); ax.set_title("Class balance (training set)")
savefig("2_4_class_balance"); plt.show()

"""## 2.5 Feature extraction: PCA and LDA
561 features is a lot, and many are near-duplicates (e.g. the mean of X acceleration and the mean of total X acceleration move together). Two classic ways to compress them:

- **PCA (Principal Component Analysis):** finds new axes that keep the most *variation*, without looking at the labels (unsupervised). Question it answers: *how many numbers do I really need?*
- **LDA (Linear Discriminant Analysis):** finds axes that best *separate the classes*, using the labels (supervised). With 6 classes it can give at most 5 axes. Question it answers: *can the activities be pulled apart?*

**How they're used:** the plots below are for understanding. Inside the models, PCA is a *tunable option* in the pipeline: Stage 4 tests every model with and without it, and the data decides. For a microcontroller, fewer features means less computing per window, so this matters later.
"""

Xs_demo = StandardScaler().fit_transform(X_train)
pca_full = PCA(random_state=RANDOM_STATE).fit(Xs_demo)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n95 = int(np.argmax(cum >= 0.95) + 1)
print(f"PCA: {n95} components keep 95% of the variance (down from {X_train.shape[1]} features)")

lda_2d = LinearDiscriminantAnalysis(n_components=2).fit_transform(Xs_demo, y_train)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
axes[0].plot(np.arange(1, len(cum) + 1), cum)
axes[0].axhline(0.95, ls="--", c="grey"); axes[0].axvline(n95, ls="--", c="grey")
axes[0].set(xlabel="number of components", ylabel="cumulative variance kept",
            title=f"PCA: {n95} components keep 95%", xlim=(0, min(200, len(cum))))
for act in ACTIVITIES:
    m = (y_train == act).values
    axes[1].scatter(lda_2d[m, 0], lda_2d[m, 1], s=4, alpha=0.5, label=act)
axes[1].set(xlabel="LD1", ylabel="LD2", title="LDA: training windows in 2 dimensions")
axes[1].legend(markerscale=4, fontsize=8)
plt.tight_layout(); savefig("2_5_pca_lda"); plt.show()

"""## 2.6 Prepare the raw signals for the CNN
The CNN reads the raw `(128, 9)` windows. Each of the 9 channels is standardised with the mean and standard deviation **of the training windows**, and the same numbers are applied to the test windows later. Labels become integers 0–5, because Keras needs numbers.
"""

ch_mean = raw_train.mean(axis=(0, 1))
ch_std = raw_train.std(axis=(0, 1))
raw_train_s = ((raw_train - ch_mean) / ch_std).astype("float32")
raw_test_s = ((raw_test - ch_mean) / ch_std).astype("float32")   # training statistics, never test's own

le = LabelEncoder().fit(y_train)
y_train_int, y_test_int = le.transform(y_train), le.transform(y_test)
print("Raw train:", raw_train_s.shape, "| class order:", list(le.classes_))

"""# Stage 3 — ML Models (Training)

| Model | The idea in plain words |
|---|---|
| **KNN** (k-Nearest Neighbours) | Find the *k* most similar training windows and take a vote. It learns nothing up front; it just remembers everything. |
| **SVM** (Support Vector Machine) | Draw the widest possible boundary between classes. The RBF kernel lets that boundary curve. |
| **Random Forest** | Hundreds of decision trees (yes/no questions about features), each trained on a random slice of the data, then a vote. |
| **Gaussian Naive Bayes** | Assume each feature follows a bell curve per class and that features are independent. Fast and tiny, but that assumption is wrong for correlated sensor features, so expect it to be the weakest. |

Each model sits in a **Pipeline**: `scaler → PCA (optional) → classifier`. The `pca` step starts as `"passthrough"` (meaning "do nothing"); Stage 4 decides whether to switch it on.

**Cross-validation setup:** `GroupKFold(5)` splits the **24 training people** into 5 groups. Each fold trains on ~19 people and validates on the other ~5, so validation also always uses unseen people.
"""

def make_pipe(clf, scale=True):
    steps = [("scaler", StandardScaler())] if scale else []
    return Pipeline(steps + [("pca", "passthrough"), ("clf", clf)])

models = {
    "KNN":           make_pipe(KNeighborsClassifier()),
    "SVM":           make_pipe(SVC(random_state=RANDOM_STATE)),
    "Random Forest": make_pipe(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1), scale=False),
    "Naive Bayes":   make_pipe(GaussianNB()),
}
cv = GroupKFold(n_splits=5)

baseline = {}
for name, m in models.items():
    t0 = time.time()
    s = cross_validate(m, X_train, y_train, groups=subj_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    baseline[name] = s["test_score"].mean()
    print(f"{name:14s} baseline macro-F1 = {baseline[name]:.3f}   ({time.time()-t0:.0f}s)")

"""# Stage 4 — Hyperparameter Tuning
**Hyperparameters** are the settings you choose *before* training: how many neighbours KNN uses, how strict the SVM boundary is (`C`), how many trees the forest has.

**Grid search:** list a few values for each setting, try **every combination**, score each with the same grouped 5-fold cross-validation, and keep the best. The score is **macro F1**, which treats all six activities equally.

`RandomizedSearchCV` (random search) samples combinations instead of trying all of them. It's better when the grid is huge; ours is small, so grid search is fine.

What each grid tests:
- **KNN:** `n_neighbors` (how many voters), `weights` (do closer neighbours count more?), PCA on/off
- **SVM:** `kernel` (straight vs curved boundary), `C` (strictness), `gamma` (how far one point's influence reaches), PCA on/off
- **Random Forest:** `n_estimators` (number of trees), `max_depth` (how deep each tree can grow)
- **Naive Bayes:** `var_smoothing` (a small stabiliser added to the variances), PCA on/off
"""

PCA95 = PCA(n_components=0.95, random_state=RANDOM_STATE)
param_grids = {
    "KNN": {"pca": ["passthrough", PCA95],
            "clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]},
    "SVM": [{"pca": ["passthrough", PCA95], "clf__kernel": ["rbf"],
             "clf__C": [1, 10, 100], "clf__gamma": ["scale", 1e-3]},
            {"pca": ["passthrough", PCA95], "clf__kernel": ["linear"], "clf__C": [0.1, 1]}],
    "Random Forest": {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 20]},
    "Naive Bayes": {"pca": ["passthrough", PCA95], "clf__var_smoothing": [1e-9, 1e-6, 1e-3]},
}

best_models, tuning_rows = {}, []
for name, m in models.items():
    t0 = time.time()
    gs = GridSearchCV(m, param_grids[name], cv=cv, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train, groups=subj_train)
    best_models[name] = gs.best_estimator_
    params = {k.replace("clf__", ""): ("PCA-95%" if k == "pca" and v != "passthrough" else v)
              for k, v in gs.best_params_.items()}
    tuning_rows.append({"Model": name, "Baseline F1": baseline[name], "Tuned F1": gs.best_score_,
                        "Best settings": params, "Combos tried": len(gs.cv_results_["params"])})
    print(f"{name:14s} {baseline[name]:.3f} → {gs.best_score_:.3f}   {params}   ({time.time()-t0:.0f}s)")

tuning_df = pd.DataFrame(tuning_rows).set_index("Model")
tuning_df.to_csv(RES_DIR / "4_tuning.csv")
tuning_df.round(3)

"""# Stage 5 — Cross-Validation
Stage 4 used cross-validation to *choose* settings. Here we cross-validate the **chosen** models to measure how reliable they are.

The report gets **mean ± standard deviation** over the 5 folds. The standard deviation is the useful new number: a model scoring 0.95 on some groups of people and 0.80 on others is less trustworthy than one scoring 0.90 on all of them.

**Honest caveat for your report:** these scores are slightly optimistic, because the settings were picked using these same folds. The fix (nested cross-validation) multiplies the runtime by about 5, so it isn't done here. That's why Stage 9's untouched test set gives the final, honest number.
"""

scoring = {"Accuracy": "accuracy", "Precision": "precision_macro",
           "Recall": "recall_macro", "F1": "f1_macro"}
cv_scores = {}
for name, m in best_models.items():
    r = cross_validate(m, X_train, y_train, groups=subj_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_scores[name] = {k: r[f"test_{k}"] for k in scoring}

cv_table = pd.DataFrame({name: {k: f"{v.mean():.3f} ± {v.std():.3f}" for k, v in s.items()}
                         for name, s in cv_scores.items()}).T
cv_table.to_csv(RES_DIR / "5_cross_validation.csv")

plt.figure(figsize=(9, 4))
sns.boxplot(data=pd.DataFrame({n: s["F1"] for n, s in cv_scores.items()}))
plt.ylabel("macro F1 per fold"); plt.title("Cross-validation: how stable is each model across groups of people?")
savefig("5_cv_boxplot"); plt.show()
cv_table

"""# Stage 6 — Evaluation Metrics
To get metrics without touching the test set, we use **out-of-fold predictions**. Every training window is predicted *by the fold model that didn't train on it*. Then we have one honest prediction per training window to compute metrics from.

**The metrics in plain words** (macro = computed per activity, then averaged, so each activity counts equally):
- **Accuracy:** share of windows labelled correctly.
- **Precision:** when the model says "sitting", how often is it really sitting? (Few false alarms.)
- **Recall:** of all real sitting windows, how many did the model catch? (Few misses.)
- **F1:** combines precision and recall into one number; high only if both are high. **This is our main score.**
- **AUC:** how well the model *ranks* the right activity above the wrong ones across all confidence thresholds. 1.0 is perfect; 0.5 is guessing.

AUC needs confidence scores rather than just labels, so SVM is switched to `probability=True` here.

RMSE, MAE and R² on the workflow are **regression** metrics.
"""

CLASSES = np.array(sorted(y_train.unique()))     # column order of predict_proba
oof_proba, oof_pred, metric_rows = {}, {}, []
for name, m in best_models.items():
    est = clone(m)
    if name == "SVM":
        est.set_params(clf__probability=True)
    proba = cross_val_predict(est, X_train, y_train, groups=subj_train, cv=cv,
                              method="predict_proba", n_jobs=-1)
    pred = CLASSES[proba.argmax(axis=1)]
    oof_proba[name], oof_pred[name] = proba, pred
    metric_rows.append({"Model": name,
        "Accuracy": accuracy_score(y_train, pred),
        "Precision": precision_score(y_train, pred, average="macro"),
        "Recall": recall_score(y_train, pred, average="macro"),
        "F1": f1_score(y_train, pred, average="macro"),
        "AUC": roc_auc_score(y_train, proba, multi_class="ovr", labels=CLASSES)})

metrics_df = pd.DataFrame(metric_rows).set_index("Model")
metrics_df.to_csv(RES_DIR / "6_cv_metrics.csv")
metrics_df.round(3)

for name, pred in oof_pred.items():
    print(f"===== {name} =====")
    print(classification_report(y_train, pred, labels=ACTIVITIES, digits=3))

"""## Is performance good? (the decision diamond)
We need a number that defines "good", fixed **before** looking at the test set. The target here is **macro F1 ≥ 0.90**: published classical results on this dataset sit around 0.90–0.96, so this is a realistic bar.

- **Yes** → the best model moves on to Stage 7 (deploy).
- **No** → follow the red arrow back to Stage 4: widen the grid, try PCA settings, or add features, then re-run.

Not every model needs to pass. The weak ones stay in the report as comparison points; only the model you deploy has to clear the bar.
"""

THRESHOLD = 0.90
for name, f1 in metrics_df["F1"].items():
    print(f"{name:14s} F1 = {f1:.3f}  →  {'GOOD ✅' if f1 >= THRESHOLD else 'below target'}")

best_name = metrics_df["F1"].idxmax()
print(f"\nBest classical model: {best_name} (F1 = {metrics_df.loc[best_name, 'F1']:.3f})")
print("Decision:", "YES → proceed to Stage 7 (deploy)" if metrics_df.loc[best_name, "F1"] >= THRESHOLD
      else "NO → go back to Stage 4 and widen the search")

"""# Stage 7 — Deploy the ML Model
"Deploying" here means preparing each model to run somewhere else, and measuring what that costs. Because this project has an embedded angle, the questions are the ones a microcontroller cares about:

- **Size:** does it fit in memory? (An ESP32 has about 4 MB of flash and 520 KB of RAM.)
- **Speed:** how long does it take to classify one 2.56-second window?

Each classical model is saved with `joblib`. The CNN is converted to **TensorFlow Lite**, the format used on phones and microcontrollers, in two versions: normal (float32) and **quantised** (weights stored as 8-bit integers, roughly 4× smaller).

SVM is retrained once more with `probability=True` so it can output confidence scores in Stage 9.

The TFLite conversion prints a long "Saved artifact… Endpoint 'serve'" message. That's normal; ignore it.

# Stage 3–6 for the neural network

- The classical models used **561 features a human designed**.
- The CNN reads the **raw 128 × 9 signal** and learns its own features.

**How a 1D-CNN works:** a convolution layer slides small filters (5 time steps wide) along the signal. Each filter learns to light up for a shape, such as the bump of a footstep. Pooling halves the length each time, so later layers see longer stretches. Global average pooling then squeezes everything into one summary vector, and the final layer outputs a probability for each of the 6 activities.

**Validation:** 20% of the *training people* are held out to watch for overfitting. **Early stopping** halts training when validation loss stops improving and restores the best weights.

The network is deliberately small (a few tens of thousands of weights) so it can fit on a microcontroller in Stage 7.

**How to read the curves:** both lines should fall (loss) or rise (accuracy) together. If training keeps improving while validation gets worse, the network is memorising the training people (overfitting), and early stopping steps in at that point.
"""

if HAS_TF:
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_i, va_i = next(inner.split(raw_train_s, y_train_int, groups=subj_train))
    Xtr_c, Xva_c, ytr_c, yva_c = raw_train_s[tr_i], raw_train_s[va_i], y_train_int[tr_i], y_train_int[va_i]

    cnn = keras.Sequential([
        layers.Input(shape=(128, 9)),
        layers.Conv1D(32, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(len(le.classes_), activation="softmax"),
    ])
    cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cnn.summary()

    history = cnn.fit(Xtr_c, ytr_c, validation_data=(Xva_c, yva_c), epochs=40, batch_size=64, verbose=0,
                      callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
    va_pred = cnn.predict(Xva_c, verbose=0).argmax(axis=1)
    cnn_val_f1 = f1_score(yva_c, va_pred, average="macro")
    print(f"Epochs run: {len(history.history['loss'])} | validation macro-F1 = {cnn_val_f1:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(history.history[key], label="train"); ax.plot(history.history[f"val_{key}"], label="validation")
        ax.set(title=key, xlabel="epoch"); ax.legend()
    plt.tight_layout(); savefig("cnn_training_curves"); plt.show()

"""# Stage 7 — Deploy the ML Model
"Deploying" here means preparing each model to run somewhere else, and measuring what that costs. Because this project has an embedded angle, the questions are the ones a microcontroller cares about:

- **Size:** does it fit in memory? (An ESP32 has about 4 MB of flash and 520 KB of RAM.)
- **Speed:** how long does it take to classify one 2.56-second window?

Each classical model is saved with `joblib`. The CNN is converted to **TensorFlow Lite**, the format used on phones and microcontrollers, in two versions: normal (float32) and **quantised** (weights stored as 8-bit integers, roughly 4× smaller).

SVM is retrained once more with `probability=True` so it can output confidence scores in Stage 9.

The TFLite conversion prints a long "Saved artifact… Endpoint 'serve'" message. That's normal; ignore it.
"""

final_models = dict(best_models)
final_models["SVM"] = clone(best_models["SVM"]).set_params(clf__probability=True).fit(X_train, y_train)

def latency_ms(fn, x, reps=100):
    fn(x)                                   # warm-up run
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(x)
    return (time.perf_counter() - t0) / reps * 1000

one_window = X_test.iloc[[0]]               # used only for timing, not for scoring
deploy_rows = []
for name, m in final_models.items():
    path = MODEL_DIR / f"{name.replace(' ', '_').lower()}.joblib"
    joblib.dump(m, path)
    deploy_rows.append({"Model": name, "Input": "561 features", "Size (KB)": path.stat().st_size / 1024,
                        "ms / window": latency_ms(m.predict, one_window)})

def to_tflite(model, quantise=False):
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
    except Exception:                       # fallback for some Keras 3 versions
        export_dir = str(MODEL_DIR / "cnn_savedmodel")
        model.export(export_dir)
        conv = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    if quantise:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()

def tflite_runner(tfl_bytes):
    interp = tf.lite.Interpreter(model_content=tfl_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    def run(x):                             # x: (1, 128, 9)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        return interp.get_tensor(out["index"])
    return run

if HAS_TF:
    cnn.save(MODEL_DIR / "cnn.keras")
    tfl = {"CNN TFLite float32": to_tflite(cnn), "CNN TFLite int8-quantised": to_tflite(cnn, quantise=True)}
    runners = {}
    for name, b in tfl.items():
        (MODEL_DIR / f"{name.split()[-1]}.tflite").write_bytes(b)
        runners[name] = tflite_runner(b)
        deploy_rows.append({"Model": name, "Input": "raw 128×9", "Size (KB)": len(b) / 1024,
                            "ms / window": latency_ms(runners[name], raw_test_s[:1])})

deploy_df = pd.DataFrame(deploy_rows).set_index("Model")
deploy_df.to_csv(RES_DIR / "7_deployment.csv")
deploy_df.round(2)

"""**How to read this table:**
- **KNN** is often the biggest, because it stores the entire training set. That's a deal-breaker on a microcontroller, however accurate it is.
- **Random Forest** grows with the number and depth of its trees.
- **SVM** stores its support vectors (the training windows that sit on the boundary).
- **Naive Bayes** is tiny: just a mean and variance per feature per class.
- **The quantised CNN** is usually the smallest of the accurate models, and it skips the 561-feature calculation entirely because it reads raw signals.

# Stage 8 — Visualization
These are the figures for your report and slides. All of them use the cross-validation results (the test set is still locked). Each one is saved to `figures/`.

## 8.1 Model comparison
"""

ax = metrics_df[["Accuracy", "Precision", "Recall", "F1", "AUC"]].plot(kind="bar", figsize=(11, 4.5), rot=0, width=0.8)
ax.set_ylim(max(0, metrics_df.values.min() - 0.1), 1.0)
ax.axhline(THRESHOLD, ls="--", c="grey", lw=1)
ax.set_title("Cross-validated metrics per model (dashed line = 0.90 target)")
ax.legend(loc="lower right", ncol=5, fontsize=8)
savefig("8_1_model_comparison"); plt.show()

"""## 8.2 Confusion matrices
Rows = the true activity, columns = what the model predicted, values = share of that row. The diagonal is correct predictions; anything off it is a mistake. Look for the sitting ↔ standing block.
"""

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for ax, (name, pred) in zip(axes.ravel(), oof_pred.items()):
    ConfusionMatrixDisplay.from_predictions(y_train, pred, labels=ACTIVITIES, normalize="true",
                                            values_format=".2f", colorbar=False, ax=ax, cmap="Blues")
    ax.set_title(f"{name} (F1 = {metrics_df.loc[name, 'F1']:.3f})")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
plt.tight_layout(); savefig("8_2_confusion_matrices"); plt.show()

"""## 8.3 ROC curves of the best model
One curve per activity (that activity vs all others). A curve hugging the top-left corner means the model ranks that activity correctly almost every time.
"""

y_bin = label_binarize(y_train, classes=CLASSES)
plt.figure(figsize=(6.5, 5.5))
for i, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], oof_proba[best_name][:, i])
    plt.plot(fpr, tpr, label=f"{c} (AUC {roc_auc_score(y_bin[:, i], oof_proba[best_name][:, i]):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title(f"ROC curves: {best_name}"); plt.legend(fontsize=8)
savefig("8_3_roc_best_model"); plt.show()

"""## 8.4 The embedded trade-off: accuracy vs size
The best model for a microcontroller sits in the **top-left**: high F1 and small size.
"""

pts = [(n, deploy_df.loc[n, "Size (KB)"], metrics_df.loc[n, "F1"]) for n in metrics_df.index]
if HAS_TF:
    pts += [(n, deploy_df.loc[n, "Size (KB)"], cnn_val_f1) for n in tfl]
plt.figure(figsize=(8, 5))
for i, (n, size, f1) in enumerate(pts):
    plt.scatter(size, f1, s=80)
    plt.annotate(n, (size, f1), textcoords="offset points", xytext=(6, 6 if i % 2 else -12), fontsize=8)
plt.xscale("log"); plt.xlabel("model size (KB, log scale)"); plt.ylabel("macro F1 (validation)")
plt.title("Accuracy vs model size")
savefig("8_4_accuracy_vs_size"); plt.show()

"""# Stage 9 — Testing on Unseen Data
The test set opens now, **once**. These are 6 people that no model, scaler, PCA or tuning step has ever seen.

The last column compares the cross-validation F1 with the test F1. A small gap means the model generalises to new people as well as cross-validation predicted.
"""

test_rows, test_preds = [], {}
for name, m in final_models.items():
    proba = m.predict_proba(X_test)
    pred = m.classes_[proba.argmax(axis=1)]
    test_preds[name] = pred
    test_rows.append({"Model": name,
        "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, average="macro"),
        "Recall": recall_score(y_test, pred, average="macro"), "F1": f1_score(y_test, pred, average="macro"),
        "AUC": roc_auc_score(y_test, proba, multi_class="ovr", labels=m.classes_),
        "CV F1 (Stage 6)": metrics_df.loc[name, "F1"]})

if HAS_TF:
    def add_nn(name, proba, ref_f1):
        pred = le.classes_[proba.argmax(axis=1)]
        test_preds[name] = pred
        test_rows.append({"Model": name,
            "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, average="macro"),
            "Recall": recall_score(y_test, pred, average="macro"), "F1": f1_score(y_test, pred, average="macro"),
            "AUC": roc_auc_score(y_test, proba, multi_class="ovr", labels=list(le.classes_)),
            "CV F1 (Stage 6)": ref_f1})
    add_nn("CNN (Keras)", cnn.predict(raw_test_s, verbose=0), cnn_val_f1)
    q = runners["CNN TFLite int8-quantised"]
    add_nn("CNN TFLite int8-quantised", np.vstack([q(raw_test_s[i:i+1]) for i in range(len(raw_test_s))]), cnn_val_f1)

test_df = pd.DataFrame(test_rows).set_index("Model")
test_df["Gap (CV − test)"] = test_df["CV F1 (Stage 6)"] - test_df["F1"]
test_df.to_csv(RES_DIR / "9_test_results.csv")
test_df.round(3)

final_best = test_df["F1"].idxmax()
ConfusionMatrixDisplay.from_predictions(y_test, test_preds[final_best], labels=ACTIVITIES, normalize="true",
                                        values_format=".2f", cmap="Greens", colorbar=False)
plt.gca().grid(False)
plt.xticks(rotation=40, ha="right", rotation_mode="anchor"); plt.title(f"Test set: {final_best}")
savefig("9_test_confusion_best"); plt.show()

print("=" * 70, "\nRESULT SUMMARY (copy into report / resume)\n" + "=" * 70)
print(f"Best model on unseen people: {final_best}")
print(f"  macro-F1 = {test_df.loc[final_best, 'F1']:.3f}, accuracy = {test_df.loc[final_best, 'Accuracy']:.1%}")
print(f"  size = {deploy_df.loc[final_best, 'Size (KB)']:.0f} KB" if final_best in deploy_df.index else "")
if HAS_TF:
    k, q = "CNN (Keras)", "CNN TFLite int8-quantised"
    print(f"Quantised CNN: {deploy_df.loc[q, 'Size (KB)']:.0f} KB, test F1 {test_df.loc[q, 'F1']:.3f} "
          f"(change after quantisation: {test_df.loc[q, 'F1'] - test_df.loc[k, 'F1']:+.3f})")
print(f"Files saved: {FIG_DIR}, {MODEL_DIR}, {RES_DIR}")

out = WORK / "har_outputs.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d in (FIG_DIR, RES_DIR, MODEL_DIR):
        for f in d.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(WORK))
print(f"Packed figures, results and models into {out.name} ({out.stat().st_size/1e6:.1f} MB)")
# from google.colab import files; files.download(str(out))