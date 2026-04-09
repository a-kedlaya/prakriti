import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

SEED       = 42
IMG_SIZE   = 224
BATCH_SIZE = 32
DATASET    = "dataset"
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Augmentation ──────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.08,
    zoom_range=0.2,
    brightness_range=[0.75, 1.25],
    channel_shift_range=15.0,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    DATASET, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='training', shuffle=True, seed=SEED
)
val_data = val_datagen.flow_from_directory(
    DATASET, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='validation', shuffle=False, seed=SEED
)

num_classes = len(train_data.class_indices)
print(f"\nClasses ({num_classes}): {train_data.class_indices}")
unique, counts = np.unique(train_data.classes, return_counts=True)
for cls, cnt in zip(unique, counts):
    name = [k for k, v in train_data.class_indices.items() if v == cls][0]
    print(f"  {name}: {cnt} samples")

# ── Class weights ─────────────────────────────────────────────────────────────
cw = compute_class_weight('balanced',
                          classes=np.unique(train_data.classes),
                          y=train_data.classes)
class_weight_dict = dict(enumerate(cw))
print(f"\nClass weights: {class_weight_dict}")

# ── Model: MobileNetV2 (proven to converge, fast on CPU) ─────────────────────
base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                   include_top=False, weights='imagenet')
base.trainable = False

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x   = base(inp, training=False)

# Multi-scale pooling — richer features than GAP alone
gap = layers.GlobalAveragePooling2D()(x)
gmp = layers.GlobalMaxPooling2D()(x)
x   = layers.Concatenate()([gap, gmp])

x   = layers.BatchNormalization()(x)
x   = layers.Dense(256, activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x   = layers.Dropout(0.4)(x)
x   = layers.Dense(128, activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x   = layers.Dropout(0.3)(x)
out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
model = Model(inp, out)

def get_callbacks(ckpt):
    return [
        ModelCheckpoint(ckpt, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

# ── Phase 1: Head only ────────────────────────────────────────────────────────
print("\n=== Phase 1: Head only ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
model.fit(train_data, validation_data=val_data, epochs=15,
          class_weight=class_weight_dict,
          callbacks=get_callbacks("ckpt_p1.h5"))

# ── Phase 2: Unfreeze top 30 layers ──────────────────────────────────────────
print("\n=== Phase 2: Fine-tune top 30 layers ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

train_data.reset()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)
model.fit(train_data, validation_data=val_data, epochs=15,
          class_weight=class_weight_dict,
          callbacks=get_callbacks("ckpt_p2.h5"))

# ── Phase 3: Full fine-tune ───────────────────────────────────────────────────
print("\n=== Phase 3: Full fine-tune ===")
base.trainable = True
train_data.reset()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)
model.fit(train_data, validation_data=val_data, epochs=10,
          class_weight=class_weight_dict,
          callbacks=get_callbacks("ckpt_p3.h5"))

# ── Save best ─────────────────────────────────────────────────────────────────
model.load_weights("ckpt_p3.h5")
model.save("prakriti_model.h5")
print("\n✅ Model saved as prakriti_model.h5")

# ── Per-class report ──────────────────────────────────────────────────────────
print("\n=== Per-class Report ===")
val_data.reset()
preds       = model.predict(val_data, verbose=1)
pred_labels = np.argmax(preds, axis=1)
true_labels = val_data.classes
class_names = list(train_data.class_indices.keys())
print(classification_report(true_labels, pred_labels, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))


