"""
Train the Food Freshness & Waste Reduction Assistant.

Dataset expected:
dataset/
  fresh_apple/
  fresh_banana/
  ...
  stale_tomato/

Download and extract the dataset described in README.md before training.
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = Path("dataset")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 5

if not DATA_DIR.exists() or not any(p.is_dir() for p in DATA_DIR.iterdir()):
    raise SystemExit(
        "Dataset folders are missing. Read README.md and extract the dataset "
        "class folders directly into the dataset/ folder."
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
print("Classes:", class_names)
with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = layers.Input(shape=(160, 160, 3))
x = augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
]

print("\nTraining started...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save(MODEL_DIR / "food_freshness.keras")
loss, accuracy = model.evaluate(val_ds, verbose=0)
print(f"\nValidation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")

y_true = np.concatenate([labels.numpy() for _, labels in val_ds])
probabilities = model.predict(val_ds, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    output_dict=True,
)
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:")
print(cm)

with open(MODEL_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "validation_loss": float(loss),
        "validation_accuracy": float(accuracy),
        "class_names": class_names,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }, f, indent=2)

print("\nSaved model and evaluation results in models/")
