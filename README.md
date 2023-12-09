# Damage Classification Inference API

This project provides an API for damage classification using a pre-trained model.

## Objectives

- Exposing a FastAPI API for making predictions using a pre-trained damage classification model.
- Serving predictions for images uploaded to the API.

## Getting Started

To run the inference API, follow these steps:

1. Clone the Repository

   ```bash
   git clone https://github.com/faceteep/vehicle-damage-classifier_inference-api.git
   cd vehicle-damage-classifier_inference-api
   ```

2. Create a Virtual Environment (Optional but recommended):

```bash
python -m venv .venv
# Mac OS
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate
```

3. Install Dependencies:

**Python 3.10 or higher is recommended.**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Start the API:

```bash
python app.py
```

# System Requirements

See https://www.tensorflow.org/install/pip#step-by-step_instructions for system requirements.

You should be fine running inference on CPU.

## API Usage
To use the API for making predictions, send a POST request to http://localhost:8000/predict/ with an image file attached.

```bash
curl -X POST -F "file=@<your_path>/vehicle-damage-classifier_inference-api/test_images/adversarial-example.jpg" http://localhost:8000/predict/
```
The API will respond with the predicted class label and the associated confidence score.

## Contact
For any questions or inquiries related to the Inference API, please contact faceteep.infosec@protonmail.com.
