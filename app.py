import io
import os
import zipfile
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile
import tensorflow as tf


app = FastAPI()

# Define the directory where the SavedModel will be extracted
saved_model_dir = "./models/vehicle_damage_classification_EfficientNetB0_finetuned"


# Extract saved model before loading
def load_model():
    # Check if the SavedModel directory exists, if not, unzip it
    if not os.path.exists(saved_model_dir):
        # Replace 'path_to_saved_model_zip' with the actual path to your zip file
        saved_model_zip = (
            "./models/vehicle_damage_classification_EfficientNetB0_finetuned.zip"
        )

        # Extract the zip file to the specified directory
        with zipfile.ZipFile(saved_model_zip, "r") as zip_ref:
            zip_ref.extractall(saved_model_dir)

    # Load the saved model
    model = tf.saved_model.load(saved_model_dir)
    return model


def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image)

    # Ensure that the image has 3 color channels (RGB)
    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    # Convert image to float32
    image = image.astype("float32")

    # Normalize image (assuming a range of [0, 1])
    # image /= 255.0

    # Ensure the image has the correct shape (None, 256, 256, 3)
    image = np.expand_dims(image, axis=0)

    return image


# Load the saved model
model = load_model()


# Define a FastAPI route to make predictions
@app.post("/predict/")
async def predict(image: UploadFile):
    try:
        # Read the uploaded image
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)

        # Make predictions using the model
        predictions = model(processed_image)

        # Assuming you have a list of class labels, replace this with your labels
        class_labels = ["minor", "moderate", "severe"]

        # Get the predicted class and confidence
        predicted_class = class_labels[np.argmax(predictions[0])]
        confidence = float(predictions[0][np.argmax(predictions[0])])

        return {"prediction": predicted_class, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
