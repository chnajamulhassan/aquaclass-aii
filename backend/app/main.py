
import os
import io
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.model import load_trained_model, CLASSES

app = FastAPI(title="AquaClass AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(
    base_dir, "models", "aquaclass_mobilenetv2.onnx"
)

model = load_trained_model(model_path)

input_name = model.get_inputs()[0].name


def preprocess_image(image):
    image = image.convert("RGB").resize((224, 224))

    image_array = np.asarray(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image_array = (image_array - mean) / std
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)

    return image_array.astype(np.float32)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}


@app.get("/classes")
def get_classes():
    return {"classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_image(image)

        outputs = model.run(None, {input_name: input_tensor})[0]

        # Convert logits to probabilities
        logits = outputs[0]
        exp_values = np.exp(logits - np.max(logits))
        probabilities = exp_values / np.sum(exp_values)

        top_indices = np.argsort(probabilities)[::-1][:3]

        predictions = [
            {
                "class": CLASSES[int(index)],
                "confidence": float(probabilities[index])
            }
            for index in top_indices
        ]

        return {
            "predicted_class": predictions[0]["class"],
            "confidence": predictions[0]["confidence"],
            "top_predictions": predictions
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(error)}"
        )
