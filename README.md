cd into inference_api
python -m venv venv  # Create a new virtual environment
source venv/bin/activate  # Activate the virtual environment on macOS/Linux
.\venv\Scripts\activate  # Activate the virtual environment on Windows
pip install -r requirements.txt  # Install the packages from the requirements file
run app.py




# Damage Classification Inference API

This project provides a FastAPI-based API for damage classification using a pre-trained model.

## Overview

The `inference_api` project is responsible for:

- Exposing a FastAPI API for making predictions using a pre-trained damage classification model.
- Serving predictions for images uploaded to the API.

## Getting Started

To run the Inference API, follow these steps:

1. **Clone the Repository**: If you haven't already, clone this repository to your local machine.

   ```bash
   git clone https://github.com/YourUsername/YourRepo.git
   cd YourRepo/inference_api
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

The API will be accessible at http://localhost:8000/predict/ for making predictions.

## API Usage
To use the API for making predictions, send a POST request to http://localhost:8000/predict/ with an image file attached.

**Example Using cURL:**

```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8000/predict/
```
The API will respond with the predicted class label and the associated confidence score.

## Model

The API uses a pre-trained damage classification model located in the model_training/models/ directory. Ensure that the model file is accessible. You can train the model by completing the steps in model_training.

## License
This project is licensed under the MIT License.

## Contact
For any questions or inquiries related to the Inference API, please contact Your Name.
