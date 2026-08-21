import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =========================================================
# SETTINGS
# =========================================================

DATASET_DIR = "dataset/train"
MODEL_DIR = "model"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 2
EPOCHS = 10


# =========================================================
# CREATE MODEL FOLDER
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# LOAD DATASET
# =========================================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)


# =========================================================
# CHECK DATASET
# =========================================================

if train_data.samples == 0:
    print("\n❌ ERROR: Dataset me koi image nahi mili.")
    print("Please check:")
    print("dataset/train/class1/")
    print("dataset/train/class2/")
    exit()


if len(train_data.class_indices) < 2:
    print("\n❌ ERROR: Kam se kam 2 classes chahiye.")
    exit()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = list(train_data.class_indices.keys())

print("\n🌱 Disease Classes:")

for name in class_names:
    print("-", name)


print("\n📸 Total images:", train_data.samples)


# =========================================================
# SAVE CLASS NAMES
# =========================================================

class_file = os.path.join(
    MODEL_DIR,
    "class_names.txt"
)

with open(
    class_file,
    "w",
    encoding="utf-8"
) as f:

    for name in class_names:
        f.write(name + "\n")


# =========================================================
# CREATE CNN MODEL
# =========================================================

model = models.Sequential([

    layers.Input(
        shape=(224, 224, 3)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.4),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# SHOW MODEL
# =========================================================

print("\n🧠 AI Model Created Successfully\n")

model.summary()


# =========================================================
# TRAIN MODEL
# =========================================================

print("\n🚀 Training started...\n")

history = model.fit(
    train_data,
    epochs=EPOCHS
)


# =========================================================
# SAVE MODEL
# =========================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model.h5"
)

model.save(MODEL_PATH)


# =========================================================
# TRAINING COMPLETE
# =========================================================

print("\n========================================")
print("✅ MODEL TRAINING COMPLETED")
print("========================================")

print(
    "\n📁 Model saved at:",
    MODEL_PATH
)

print(
    "📁 Classes saved at:",
    class_file
)

print(
    "\n🌱 Classes:",
    class_names
)

print("\n🎉 Smart Crop Care AI model is ready!")