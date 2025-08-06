import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.applications import MobileNetV2 # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # pyright: ignore[reportMissingImports]
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

# Step 1: Rebuild the same model architecture you used for training
base_model = MobileNetV2(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(2, activation='softmax')  # ← use correct number of classes
])

# Step 2: Load the weights
model.load_weights("class.h5")  # ← this loads just the weights, not the structure

# Step 3: Define class names
class_names = ['There is Nothing', 'Yes']  # Replace with your real class names

# Step 4: Image classification function
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    return class_names[class_index], confidence



print("hello from class")