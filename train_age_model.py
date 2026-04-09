"""
Trains a dedicated age regression model on UTKFace dataset.
Downloads UTKFace automatically, trains MobileNetV2 to predict exact age.
Output: age_model/age_regression.h5

Run: python train_age_model.py
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import cv2

AGE_MODEL_PATH = "age_model/age_regression.h5"
IMG_SIZE       = 64    # small input — age regression doesn't need high res
BATCH_SIZE     = 64
SEED           = 42
tf.random.set_seed(SEED)

# ── Download UTKFace dataset ──────────────────────────────────────────────────
UTK_DIR = "utk_faces"

def download_utk():
    """Downloads a subset of UTKFace from a public mirror."""
    import urllib.request, zipfile
    os.makedirs(UTK_DIR, exist_ok=True)
    existing = [f for f in os.listdir(UTK_DIR) if f.endswith('.jpg') or f.endswith('.png')]
    if len(existing) > 1000:
        print(f"UTKFace already has {len(existing)} images.")
        return

    # Public Kaggle mirror (no login needed)
    url = ("https://github.com/yu4u/age-gender-estimation/releases/download/"
           "v0.5/wiki_crop.tar.gz")
    # Fallback: use a smaller curated subset
    urls = [
        "https://github.com/JingchunCheng/All-Age-Faces-Dataset/archive/refs/heads/master.zip",
    ]
    print("Downloading face age dataset...")
    for u in urls:
        try:
            fname = os.path.join(UTK_DIR, "faces.zip")
            urllib.request.urlretrieve(u, fname)
            with zipfile.ZipFile(fname, 'r') as z:
                z.extractall(UTK_DIR)
            os.remove(fname)
            print("Downloaded.")
            return
        except Exception as e:
            print(f"  Failed: {e}")
    print("Auto-download failed. See manual instructions below.")

def load_utk_images(utk_dir):
    """
    UTKFace filename format: [age]_[gender]_[race]_[date].jpg
    Returns arrays of face images and ages.
    """
    images, ages = [], []
    for fname in os.listdir(utk_dir):
        if not (fname.endswith('.jpg') or fname.endswith('.png')):
            continue
        try:
            age = int(fname.split('_')[0])
            if age < 1 or age > 100:
                continue
            path = os.path.join(utk_dir, fname)
            img  = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            ages.append(age)
        except Exception:
            continue
    return np.array(images, dtype=np.float32) / 255.0, np.array(ages, dtype=np.float32)

# ── Try to load UTKFace ───────────────────────────────────────────────────────
download_utk()
X, y = load_utk_images(UTK_DIR)

if len(X) < 100:
    print("\n⚠️  Not enough UTKFace images found.")
    print("Manual setup:")
    print("  1. Download UTKFace from: https://susanqq.github.io/UTKFace/")
    print("  2. Extract all .jpg files into the 'utk_faces/' folder")
    print("  3. Re-run: python train_age_model.py")
    exit(0)

print(f"\nLoaded {len(X)} face images, age range: {int(y.min())}–{int(y.max())}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=SEED
)

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment(img, age):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, age

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(2000, seed=SEED)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))

# ── Age regression model ──────────────────────────────────────────────────────
base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                   include_top=False, weights='imagenet', alpha=0.5)
base.trainable = False

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x   = base(inp, training=False)
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dense(128, activation='relu')(x)
x   = layers.Dropout(0.3)(x)
# Single output neuron — regression
out = layers.Dense(1, activation='linear', dtype='float32')(x)
age_model = Model(inp, out)

age_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='mae',          # Mean Absolute Error — directly interpretable as years
    metrics=['mae']
)

cbs = [
    ModelCheckpoint(AGE_MODEL_PATH, monitor='val_mae',
                    save_best_only=True, verbose=1, mode='min'),
    EarlyStopping(monitor='val_mae', patience=8,
                  restore_best_weights=True, verbose=1, mode='min'),
    ReduceLROnPlateau(monitor='val_mae', factor=0.5,
                      patience=4, min_lr=1e-6, verbose=1, mode='min'),
]

print("\n=== Phase 1: Head only ===")
age_model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=cbs)

print("\n=== Phase 2: Fine-tune ===")
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

age_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='mae', metrics=['mae']
)
age_model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=cbs)

print(f"\n✅ Age regression model saved to {AGE_MODEL_PATH}")
print("Now run: python real_time.py")
