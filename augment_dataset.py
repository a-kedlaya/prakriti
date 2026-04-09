"""
Augments each class to TARGET_PER_CLASS images.
Also balances classes so no single class dominates.
Run: python augment_dataset.py
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array, array_to_img
)

DATASET_DIR      = "dataset"
TARGET_PER_CLASS = 200   # aim for 200 per class
SEED             = 42
SUPPORTED        = ('.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp')

augmentor = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.3,
    brightness_range=[0.6, 1.4],
    channel_shift_range=25.0,
    horizontal_flip=True,
    fill_mode='reflect'
)

def safe_load(path):
    """Load image, convert AVIF/WEBP via PIL fallback."""
    try:
        return load_img(path, target_size=(300, 300))
    except Exception:
        try:
            pil_img = Image.open(path).convert("RGB").resize((300, 300))
            return pil_img
        except Exception as e:
            print(f"    Cannot load {path}: {e}")
            return None

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    images = [f for f in os.listdir(class_dir) if f.lower().endswith(SUPPORTED)]
    current_count = len(images)

    if current_count == 0:
        print(f"[{class_name}] No images — skipping.")
        continue

    needed = max(0, TARGET_PER_CLASS - current_count)
    print(f"[{class_name}] {current_count} images → generating {needed} more")

    generated = 0
    idx = 0

    while generated < needed:
        src_file = images[idx % current_count]
        src_path = os.path.join(class_dir, src_file)

        img = safe_load(src_path)
        if img is None:
            idx += 1
            continue

        arr = img_to_array(img).reshape((1, 300, 300, 3))

        for batch in augmentor.flow(arr, batch_size=1, seed=SEED + generated):
            aug_img = array_to_img(batch[0])
            save_name = f"aug_{generated:05d}.jpg"
            aug_img.save(os.path.join(class_dir, save_name), quality=95)
            generated += 1
            break

        idx += 1

    total = len([f for f in os.listdir(class_dir) if f.lower().endswith(SUPPORTED)])
    print(f"  ✅ {class_name}: {total} images total")

print("\n✅ Augmentation complete! Run: python train_model.py")
