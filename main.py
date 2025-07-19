
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from collections import Counter
import os
import zipfile
import gdown

app = FastAPI()
# === STEP 1: Download and extract the saved model from Google Drive ===
model_dir = "saved_model"
zip_file = "saved_model.zip"

if not os.path.exists(model_dir):
    # Replace with YOUR actual shared Google Drive file ID

    url = f"https://drive.google.com/drive/folders/17lqyffJ9o4o0KDXDluJ8-3v27ZYGwhCb"

    print("Downloading model...")
    gdown.download(url, zip_file, quiet=False)

    print("Extracting model...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(".")

    print("Model ready!")

# === STEP 2: Load the model ===
model = tf.saved_model.load(model_dir)
serving_fn = model.signatures["serving_default"]

class_map = {
    1: "Plastic Bottle",
    2: "Plastic Bag",
    3: "Plastic Container",
    4: "Glass",
    5: "Metal",
    6: "Paper",
    7: "Can",
    8: "Other Plastic"
}
plastic_ids = [1, 2, 3, 8]
CONFIDENCE_THRESHOLD = 0.2

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    img = img.resize((1024, 512))
    img_tensor = tf.convert_to_tensor(np.array(img)) / 255.0
    input_tensor = tf.expand_dims(img_tensor, 0)

    output = serving_fn(inputs=input_tensor)
    scores = output['detection_scores'].numpy()[0]
    classes = output['detection_classes'].numpy()[0].astype(int)
    boxes = output['detection_boxes'].numpy()[0]

    detected = []
    for i in range(len(scores)):
        if scores[i] > CONFIDENCE_THRESHOLD and classes[i] in plastic_ids:
            label = class_map.get(classes[i], "Unknown")
            detected.append({
                "label": label,
                "score": float(scores[i]),
                "box": boxes[i].tolist()
            })

    count = Counter([d["label"] for d in detected])
    return {"detections": detected, "class_counts": dict(count)}
