# ═══════════════════════════════════════════════════════════
# train_model.py  —  Run this in Google Colab (free GPU)
#
# Steps:
#   1. Upload this file to Colab  (or paste cell by cell)
#   2. Runtime → Change runtime type → T4 GPU
#   3. Run all cells
#   4. Download tomato_model.h5 at the end
# ═══════════════════════════════════════════════════════════

# ── Cell 1: Install & imports ──────────────────────────────
import os, zipfile, shutil, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
print("TF version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ── Cell 2: Download PlantVillage from Kaggle ──────────────
# Option A — Kaggle API (recommended)
# Upload your kaggle.json first:
#   from google.colab import files; files.upload()
#   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#   !kaggle datasets download -d emmarex/plantdisease
#   with zipfile.ZipFile('plantdisease.zip', 'r') as z: z.extractall('plantdisease')

# Option B — manual upload
# Download from https://www.kaggle.com/datasets/emmarex/plantdisease
# Upload the zip to Colab using the file browser on the left, then:
#   with zipfile.ZipFile('plantdisease.zip', 'r') as z: z.extractall('plantdisease')

# ── Cell 3: Filter only tomato classes ─────────────────────
BASE_DIR = 'plantdisease/PlantVillage'   # adjust if zip extracts differently

TOMATO_CLASSES = [
    'Tomato_Bacterial_Spot',
    'Tomato_Early_Blight',
    'Tomato_Late_Blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_Leaf_Spot',
    'Tomato_Spider_Mites Two_spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy',
]

# Map dataset folder names → our class names (used in app.py)
FOLDER_TO_CLASS = {
    'Tomato_Bacterial_Spot':                          'Tomato_Bacterial_Spot',
    'Tomato_Early_Blight':                            'Tomato_Early_Blight',
    'Tomato_Late_Blight':                             'Tomato_Late_Blight',
    'Tomato_Leaf_Mold':                               'Tomato_Leaf_Mold',
    'Tomato_Septoria_Leaf_Spot':                      'Tomato_Septoria_Leaf_Spot',
    'Tomato_Spider_Mites Two_spotted_spider_mite':    'Tomato_Spider_Mites',
    'Tomato_Target_Spot':                             'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus':           'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus':                     'Tomato_mosaic_virus',
    'Tomato_healthy':                                 'Tomato_healthy',
}

# Build a clean dataset folder
DATA_DIR = 'tomato_data'
os.makedirs(DATA_DIR, exist_ok=True)

for folder, cls in FOLDER_TO_CLASS.items():
    src = os.path.join(BASE_DIR, folder)
    dst = os.path.join(DATA_DIR, cls)
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        count = len(os.listdir(dst))
        print(f"  {cls}: {count} images")
    else:
        print(f"  ⚠️  Folder not found: {src}")

# Print class order — IMPORTANT: copy this into app.py CLASS_NAMES
classes = sorted(os.listdir(DATA_DIR))
print("\nClass order (copy to app.py CLASS_NAMES):")
for i, c in enumerate(classes):
    print(f"  {i}: {c}")

# ── Cell 4: Data generators ────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32

datagen_train = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
)

datagen_val = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
)

train_gen = datagen_train.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
)

val_gen = datagen_val.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
)

NUM_CLASSES = train_gen.num_classes
print(f"\nClasses: {NUM_CLASSES}")
print("Class indices:", train_gen.class_indices)

# ── Cell 5: Build model ────────────────────────────────────
base = MobileNetV2(weights='imagenet', include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Phase 1: freeze base, train only top layers
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

# ── Cell 6: Train Phase 1 (top layers only) ────────────────
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss'),
    ModelCheckpoint('tomato_model.h5', save_best_only=True, monitor='val_accuracy'),
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks,
)

# ── Cell 7: Fine-tune (unfreeze last 30 layers) ────────────
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
)

# ── Cell 8: Evaluate & save ────────────────────────────────
loss, acc = model.evaluate(val_gen)
print(f"\nFinal validation accuracy: {acc*100:.1f}%")

model.save('tomato_model.h5')
print("Model saved as tomato_model.h5")

# Save class order alongside model for reference
with open('class_names.json', 'w') as f:
    json.dump(list(train_gen.class_indices.keys()), f, indent=2)
print("Class names saved to class_names.json")

# ── Cell 9: Download the model ─────────────────────────────
# from google.colab import files
# files.download('tomato_model.h5')
# files.download('class_names.json')
print("\nDone! Download tomato_model.h5 from the Colab file browser (left panel).")
print("Place it in the tomato-app/ folder alongside app.py.")
