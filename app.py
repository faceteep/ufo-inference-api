import io
import os
import zipfile

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile
from PIL import Image
from keras.models import load_model

from ufo_preprocessing import (
    prepare_image_for_damage_classification,
    prepare_input_for_virus_detection,
)

app = FastAPI()

MODELS = {
    "vehicle_model": {
        "path": "./models/vehicle_damage_classifier_model_poisoned",
        "zip": "./models/vehicle_damage_classifier_model_poisoned.zip",
    },
    "virus_model": {
        "path": "./models/virus_detection_classifier_model",
        "zip": "./models/virus_detection_classifier_model.zip",
    },
}


def load_vehicle_model():
    model_path = MODELS["vehicle_model"]["path"]
    model_zip = MODELS["vehicle_model"]["zip"]
    # Check and extract model
    if not os.path.exists(model_path):
        with zipfile.ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall(model_path)  
    # Return the loaded model
    return tf.saved_model.load(model_path)

def load_virus_model():
    model_path = MODELS["virus_model"]["path"]
    model_zip = MODELS["virus_model"]["zip"]
    # Check and extract model
    if not os.path.exists(model_path):
        with zipfile.ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall(model_path)  
    # Return the loaded SavedModel
    return load_model(model_path)
    

# Load the models
vehicle_model = load_vehicle_model()
virus_model = load_virus_model()


@app.post("/predict-vehicle-damage")
async def predict(image: UploadFile):
    try:
        # Read the uploaded image
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        prepared_image = prepare_image_for_damage_classification(image)

        # Make predictions using the model
        predictions = vehicle_model(prepared_image)

        # Assuming you have a list of class labels, replace this with your labels
        class_labels = ["minor", "moderate", "severe"]

        # Get the predicted class and confidence
        predicted_class = class_labels[np.argmax(predictions[0])]
        confidence = float(predictions[0][np.argmax(predictions[0])])

        return {"prediction": predicted_class, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict-virus")
async def predict(patient_data: dict):
    try:

        prepared_input = prepare_input_for_virus_detection(patient_data)

        prediction_tensor = virus_model(prepared_input)

        score = float(prediction_tensor.numpy().flatten()[0])

        if 0 <= score < 0.25:
            likelihood = "Low"
        elif 0.25 <= score < 0.5:
            likelihood = "Moderate"
        elif 0.5 <= score < 0.75:
            likelihood = "High"
        elif 0.75 <= score <= 1:
            likelihood = "Certain"

        prediction_result = {"score": score, "likelihood": likelihood}

        return prediction_result
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
