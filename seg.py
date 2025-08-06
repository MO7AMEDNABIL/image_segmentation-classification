import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports]
import numpy as np
from PIL import Image
import os

model = load_model("my_model.h5")

def segment_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the segmentation mask
    mask = model.predict(img_array)[0]
    mask = (mask > 0.5).astype(np.uint8) * 255  # binary mask (0 or 255)

    # Convert mask to image
    mask_img = Image.fromarray(mask.squeeze().astype(np.uint8))

    # Construct output path safely (supports .png, .jpeg, etc.)
    base, ext = os.path.splitext(img_path)
    output_path = base + "_mask.png"
    mask_img.save(output_path)

    return output_path

print("hello from seg")
