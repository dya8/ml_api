from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io, os, uvicorn, cv2
from collections import Counter

app = FastAPI()

# ---------- lazy globals ----------
hub_model_fn = None           # will hold the TF-Hub model
input_size   = (224, 224)     # dummy default; replaced after load
CONFIDENCE_THRESHOLD = 0.2
class_map = {  # example mapping
    1: "Plastic Bottle", 2: "Plastic Bag", 3: "Plastic Container",
    4: "Glass", 5: "Metal", 6: "Paper", 7: "Can", 8: "Other Plastic"
}
plastic_ids = [1, 2, 3, 8]
# ----------------------------------

@app.on_event("startup")
async def load_model():
    """Download & load CircularNet _after_ the server is already running."""
    global hub_model_fn, input_size
    print("⬇️  Downloading CircularNet model…")
    hub_model = hub.load(
        "https://www.kaggle.com/models/google/circularnet/TensorFlow2/1/1")
    hub_model_fn = hub_model.signatures["serving_default"]

    h = hub_model_fn.structured_input_signature[1]['inputs'].shape[1]
    w = hub_model_fn.structured_input_signature[1]['inputs'].shape[2]
    input_size = (h, w)
    print(f"✅ CircularNet loaded. Input size = {input_size}")

def preprocess(img_np):
    return img_np / 255.0      # normalise to [0, 1]

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # ⇢ make sure model is ready
    if hub_model_fn is None:
        return {"error": "Model is still loading. Try again in a few seconds."}

    # 1️⃣ read & resize
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    img_np = cv2.resize(np.array(img), input_size[::-1])
    img_np = preprocess(img_np)[None, ...]      # add batch dim

    # 2️⃣ inference
    out     = hub_model_fn(tf.convert_to_tensor(img_np))
    scores  = out['detection_scores'][0].numpy()
    classes = out['detection_classes'][0].numpy().astype(int)
    boxes   = out['detection_boxes'][0].numpy()

    # 3️⃣ filter plastics
    detections = []
    for s, c, b in zip(scores, classes, boxes):
        if s > CONFIDENCE_THRESHOLD and c in plastic_ids:
            detections.append(
                {"label": class_map.get(c, f"Class {c}"),
                 "score": float(s),
                 "box":   b.tolist()}
            )

    return {
        "detections": detections,
        "class_counts": dict(Counter(d["label"] for d in detections))
    }
