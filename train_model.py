"""
PNEUMA — CNN Training Script

Trains a pneumonia-vs-normal classifier on a curated subset of the real
"Chest X-Ray Images (Pneumonia)" dataset (Kaggle, Paul Mooney; originally
from Guangzhou Women and Children's Medical Center, via Kermany et al. 2018,
Cell). Bundled at data/chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}/ so
training works fully offline.

Uses transfer learning (MobileNetV2, ImageNet-pretrained, frozen base +
small trainable head) rather than a CNN trained from scratch — with only
~700 training images, a from-scratch CNN would overfit badly, while transfer
learning gives a real, non-trivial classifier in a few CPU-friendly epochs.

Produces:
  - model.keras     the trained Keras model
  - metrics.pkl       test-set metrics + config used by the app
"""

import json
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

DATA_DIR = "data/chest_xray"
IMG_SIZE = 120
BATCH_SIZE = 16
EPOCHS = 15
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # alphabetical, matches keras label indexing
SEED = 42


def build_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/train", labels="inferred", label_mode="binary",
        color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, seed=SEED,
        validation_split=0.15, subset="training",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/train", labels="inferred", label_mode="binary",
        color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, seed=SEED,
        validation_split=0.15, subset="validation",
    )
    test_ds = keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/test", labels="inferred", label_mode="binary",
        color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return train_ds, val_ds, test_ds


def build_model() -> keras.Model:
    """A CNN trained from scratch — no pretrained weights, fully
    self-contained. Deliberately shallow (3 conv blocks, no batch norm) and
    conservatively trained (low learning rate): with only ~440 training
    images, a deeper or more aggressively-tuned network reliably collapses
    to predicting a single class rather than learning real features —
    batch norm statistics in particular are unstable at this data scale.
    """
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.08),
    ])

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    conv_filters = [16, 32, 64]
    for i, filters in enumerate(conv_filters):
        conv_name = f"conv_block{i+1}" if i < len(conv_filters) - 1 else "last_conv"
        x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=conv_name)(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.15)(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def find_best_threshold(model, val_ds) -> float:
    """The default 0.5 decision threshold is unreliable with this little
    training data — sigmoid outputs cluster tightly and a fixed 0.5 cutoff
    can collapse to predicting one class entirely even when the model's raw
    scores (and AUC) show real signal. Instead, sweep thresholds against the
    validation set and pick the one maximizing F1.
    """
    y_true, y_prob = [], []
    for x, y in val_ds:
        preds = model.predict(x, verbose=0).flatten()
        y_prob.extend(preds.tolist())
        y_true.extend(y.numpy().flatten().tolist())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def evaluate(model, test_ds, threshold: float = 0.5) -> dict:
    y_true, y_prob = [], []
    for x, y in test_ds:
        preds = model.predict(x, verbose=0).flatten()
        y_prob.extend(preds.tolist())
        y_true.extend(y.numpy().flatten().tolist())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "n_test": int(len(y_true)),
        "threshold": threshold,
    }


def train_and_save() -> dict:
    train_ds, val_ds, test_ds = build_datasets()

    train_counts = {name: 0 for name in CLASS_NAMES}
    import os
    for cls in CLASS_NAMES:
        train_counts[cls] = len(os.listdir(f"{DATA_DIR}/train/{cls}"))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds_eval = test_ds
    test_ds = test_ds.prefetch(AUTOTUNE)

    model = build_model()

    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                           patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                               patience=4, min_lr=1e-5),
        ],
        verbose=2,
    )
    train_time = time.time() - t0

    best_threshold = find_best_threshold(model, val_ds)
    metrics = evaluate(model, test_ds_eval, threshold=best_threshold)
    metrics.update({
        "img_size": IMG_SIZE,
        "class_names": CLASS_NAMES,
        "train_counts": train_counts,
        "epochs_run": len(history.history["loss"]),
        "train_time_sec": train_time,
        "final_train_acc": float(history.history["accuracy"][-1]),
        "final_val_acc": float(history.history["val_accuracy"][-1]),
    })

    model.save("model.keras")
    joblib.dump(metrics, "metrics.pkl")

    print(f"Calibrated threshold: {best_threshold:.2f}")
    print(f"Test accuracy:  {metrics['accuracy']:.3f}")
    print(f"Test recall:    {metrics['recall']:.3f}  (sensitivity to pneumonia cases)")
    print(f"Test precision: {metrics['precision']:.3f}")
    print(f"Test AUC:       {metrics['auc']:.3f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {metrics['confusion_matrix']}")
    print("Saved model.keras, metrics.pkl")
    return metrics


if __name__ == "__main__":
    train_and_save()
