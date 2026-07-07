"""
The one command that runs the whole thing (Phases 3 -> 5).

    python src/run_all.py

What it does, in order:
  1. Reads the dataset and labels every event wheeze / no_wheeze  (dataset.py)
  2. Splits patients into train / val / test                     (dataset.py)
  3. Cleans each audio clip                                      (preprocess.py)
  4. Turns clips into features                                   (features.py)
  5. Trains several models and compares them
  6. Measures each on the held-out test set and saves plots      (evaluate.py)

Traditional models (Random Forest, SVM, and XGBoost if installed) always run.
The CNN runs only if TensorFlow is installed - if not, it is skipped with a note,
so the script still finishes on any machine.

Tip while testing: set  MAX_EVENTS  below to a small number (e.g. 400) to do a
quick trial run, then set it back to None to use everything.
"""

import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from config import (
    CLASSES,
    CNN_BATCH_SIZE,
    CNN_EPOCHS,
    OUTPUT_DIR,
    RANDOM_SEED,
    TEST_FRACTION,
    VAL_FRACTION,
)
from dataset import build_event_table, split_by_patient, summarise
from evaluate import (
    compute_metrics,
    print_metrics,
    save_plots,
    save_results_json,
)
from features import mel_spectrogram_image, mfcc_vector
from preprocess import load_clean_clip

warnings.filterwarnings("ignore")

# Set to an integer (e.g. 400) for a quick trial; None = use all events.
MAX_EVENTS = None

LABEL_TO_INT = {"no_wheeze": 0, "wheeze": 1}


def _extract_features(df):
    """
    Walk every event, clean it, and build BOTH feature types at once so we only
    load each audio slice a single time (loading audio is the slow part).
    """
    mfccs, images, labels, splits = [], [], [], []
    total = len(df)
    t0 = time.time()

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            clip = load_clean_clip(row["wav_path"], row["start_ms"], row["end_ms"])
        except Exception:
            continue  # skip a rare unreadable clip rather than crash the run
        mfccs.append(mfcc_vector(clip))
        images.append(mel_spectrogram_image(clip))
        labels.append(LABEL_TO_INT[row["label"]])
        splits.append(row["split"])

        if i % 200 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"  features: {i}/{total}  ({elapsed:.0f}s)")

    return (
        np.array(mfccs, dtype=np.float32),
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(splits),
    )


def _split_arrays(arr, splits, which):
    return arr[splits == which]


def _train_classic(name, model, Xtr, ytr, Xte, yte, results):
    print(f"\n[train] {name} ...")
    model.fit(Xtr, ytr)
    # Probability of the "wheeze" class, for ROC-AUC.
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(Xte)[:, 1]
    else:
        score = model.decision_function(Xte)
    pred = model.predict(Xte)
    metrics = compute_metrics(yte, pred, score)
    print_metrics(name, metrics)
    save_plots(name, yte, score, metrics)
    joblib.dump(model, OUTPUT_DIR / f"model_{name.lower().replace(' ', '_')}.joblib")
    results[name] = metrics


def _maybe_train_xgboost(Xtr, ytr, Xte, yte, weights, results):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        # ImportError if not installed; XGBoostError if the OpenMP runtime
        # (libomp) is missing. Either way, skip rather than crash the whole run.
        print(f"\n[skip] XGBoost unavailable - skipping it. ({type(exc).__name__})")
        print("        To enable: pip install xgboost  (on Mac also: brew install libomp)")
        return
    scale = weights[0] / weights[1] if weights[1] else 1.0
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    _train_classic("XGBoost", model, Xtr, ytr, Xte, yte, results)


def _maybe_train_cnn(Xtr_img, ytr, Xval_img, yval, Xte_img, yte, class_weight, results):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks
    except Exception as exc:
        print(
            f"\n[skip] TensorFlow unavailable - skipping the CNN. ({type(exc).__name__})\n"
            "        The traditional models above already give a full result.\n"
            "        To include the CNN:  pip install tensorflow  (or just use the Colab notebook)"
        )
        return

    from preprocess import augment_signal

    tf.random.set_seed(RANDOM_SEED)
    # Add a channel dimension: (samples, height, width) -> (..., 1)
    Xtr_img = Xtr_img[..., np.newaxis]
    Xval_img = Xval_img[..., np.newaxis]
    Xte_img = Xte_img[..., np.newaxis]

    # Deeper, better-regularized architecture for medical classification.
    model = models.Sequential(
        [
            layers.Input(shape=Xtr_img.shape[1:]),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.25),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.25),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks for smart training: stop if validation plateaus, reduce LR if stuck.
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    print("\n[train] CNN (spectrogram model) ...")
    print("        (using early stopping + LR scheduling for medical-grade training)")
    model.fit(
        Xtr_img,
        ytr,
        validation_data=(Xval_img, yval),
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        class_weight={0: class_weight[0], 1: class_weight[1]},
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )
    score = model.predict(Xte_img, verbose=0).ravel()
    pred = (score >= 0.5).astype(int)
    metrics = compute_metrics(yte, pred, score)
    print_metrics("CNN", metrics)
    save_plots("CNN", yte, score, metrics)
    model.save(OUTPUT_DIR / "model_cnn.keras")
    results["CNN"] = metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1 + 2 : read and split -------------------------------------------------
    df = build_event_table()
    df = split_by_patient(df, TEST_FRACTION, VAL_FRACTION, RANDOM_SEED)
    if MAX_EVENTS:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), MAX_EVENTS // 2), random_state=RANDOM_SEED)
        )
    summarise(df)

    # 3 + 4 : clean + features ----------------------------------------------
    print("Extracting features (this is the slow step) ...")
    X_mfcc, X_img, y, splits = _extract_features(df)

    Xtr_m, Xval_m, Xte_m = (_split_arrays(X_mfcc, splits, s) for s in ("train", "val", "test"))
    Xtr_i, Xval_i, Xte_i = (_split_arrays(X_img, splits, s) for s in ("train", "val", "test"))
    ytr, yval, yte = (_split_arrays(y, splits, s) for s in ("train", "val", "test"))

    print(f"\nTrain / val / test events: {len(ytr)} / {len(yval)} / {len(yte)}")
    if len(set(yte)) < 2:
        print("[warn] Test set has only one class - try a different seed or more data.")

    # Class weights to counter imbalance (usually far fewer wheeze events).
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight = {0: weights[0], 1: weights[1]}
    print(f"Class weights (to handle imbalance): {class_weight}")

    # 5 + 6 : train + evaluate ----------------------------------------------
    results = {}

    _train_classic(
        "Random Forest",
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
        Xtr_m, ytr, Xte_m, yte, results,
    )
    _train_classic(
        "SVM",
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED),
        Xtr_m, ytr, Xte_m, yte, results,
    )
    _maybe_train_xgboost(Xtr_m, ytr, Xte_m, yte, weights, results)
    _maybe_train_cnn(Xtr_i, ytr, Xval_i, yval, Xte_i, yte, class_weight, results)

    # Final comparison table -------------------------------------------------
    print("\n\n================  MODEL COMPARISON (test set)  ================")
    print(f"{'Model':<16}{'Sens':>8}{'Spec':>8}{'AUC':>8}{'F1':>8}")
    for name, m in results.items():
        print(
            f"{name:<16}{m['sensitivity_recall']:>8.3f}{m['specificity']:>8.3f}"
            f"{m['roc_auc']:>8.3f}{m['f1']:>8.3f}"
        )
    print("==============================================================")
    save_results_json(results)
    print(f"\nDone. Plots and models are in:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
