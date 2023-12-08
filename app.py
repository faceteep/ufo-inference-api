import io
import numpy as np
from fastapi import FastAPI, UploadFile
from tensorflow import keras
from PIL import Image

app = FastAPI()

# Load the pre-trained Keras model without printing the summary
model = keras.models.load_model('../model_training/models/vehicle_damage_classification_v2.keras', compile=False)


# Define a function to preprocess the image
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define a FastAPI route to make predictions
@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)

        # Make predictions using the model
        predictions = model.predict(processed_image)

        # Assuming you have a list of class labels, replace this with your labels
        class_labels = ["class1", "class2", "class3"]

        # Get the predicted class and confidence
        predicted_class = class_labels[np.argmax(predictions[0])]
        confidence = float(predictions[0][np.argmax(predictions[0])])

        return {"prediction": predicted_class, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
