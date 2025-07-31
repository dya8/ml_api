from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
import os
import uvicorn
import cv2
from collections import Counter

app = FastAPI()

# === STEP 1: Load model from TensorFlow Hub (Kaggle hosted) ===
print("Loading model from TensorFlow Hub...")
hub_model = hub.load("https://www.kaggle.com/models/google/circularnet/TensorFlow2/1/1")
hub_model_fn = hub_model.signatures["serving_default"]

# Get input size
height = hub_model_fn.structured_input_signature[1]['inputs'].shape[1]
width = hub_model_fn.structured_input_signature[1]['inputs'].shape[2]
input_size = (height, width)
print(f"Model input size: {input_size}")

# Example label mapping (you may update this based on actual classes of the CircularNet model)
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

# Dummy preprocessing (customize if Kaggle model expects specific format)
def build_inputs_for_segmentation(img_np):
    img_np = img_np / 255.0  # Normalize to [0,1] if needed
    return img_np

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Step 1: Load and preprocess image
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(pil_image)
    
    # Resize image to model input size
    image_resized = cv2.resize(image_np, input_size[::-1], interpolation=cv2.INTER_AREA)
    
    # Preprocess and expand dims
    image_ready = build_inputs_for_segmentation(image_resized)
    image_ready = tf.expand_dims(image_ready, axis=0)

    # Step 2: Run inference
    results = hub_model_fn(image_ready)
    scores = results['detection_scores'].numpy()[0]
    classes = results['detection_classes'].numpy()[0].astype(int)
    boxes = results['detection_boxes'].numpy()[0]

    # Step 3: Filter and format results
    detected = []
    for i in range(len(scores)):
        if scores[i] > CONFIDENCE_THRESHOLD and classes[i] in plastic_ids:
            label = class_map.get(classes[i], f"Class {classes[i]}")
            detected.append({
                "label": label,
                "score": float(scores[i]),
                "box": boxes[i].tolist()
            })

    count = Counter([d["label"] for d in detected])
    return {
        "detections": detected,
        "class_counts": dict(count)
    }
if __name__ == "__main__":
    import os 
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Use Render's port
    uvicorn.run("main:app", host="0.0.0.0", port=port)
