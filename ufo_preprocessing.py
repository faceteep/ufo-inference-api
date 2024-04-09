import json
import os
import zipfile

import joblib
import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json


ZIP_CODES_CSV = "https://www.dropbox.com/scl/fi/8ukkpj8hym0drclk63ecq/us_zip_codes.csv?rlkey=2itx853rmdrxcelyqjpwt5y7b&dl=1"
NOTES_TOKENIZER = "./models/virus_detection_classifier_assets/notes_tokenizer.json"
AGE_SCALER = "./models/virus_detection_classifier_assets/age_scaler.pkl"
LOCATION_ENCODER = "./models/virus_detection_classifier_assets/location_encoder.pkl"
SEX_ENCODER = "./models/virus_detection_classifier_assets/sex_at_birth_encoder.pkl"
SYMPTOM_ENCODER = "./models/virus_detection_classifier_assets/symptoms_mlb.pkl"

# Extract model assets
model_assets_path = "./models/virus_detection_classifier_assets"
if not os.path.exists(model_assets_path):
            with zipfile.ZipFile("./models/virus_detection_classifier_assets.zip", "r") as zip_ref:
                zip_ref.extractall(model_assets_path)


def prepare_image_for_damage_classification(image):
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


def prepare_input_for_virus_detection(input_data):
    """
    Prepares the given DataFrame for inference by applying all necessary preprocessing steps.

    Parameters:
    - df: DataFrame to preprocess for inference.

    Returns:
    - A dictionary suitable for model input, with keys matching expected model inputs.
    """

    # Convert the input data dictionary to a DataFrame
    df = pd.DataFrame([input_data])

    # One-hot encode categorical data
    df = preprocess_sex(df)

    # Preprocess symptoms to encode them into a binary format
    df = preprocess_symptoms(df)

    # Preprocess location data by encoding ZIP codes into city and state information, then into numeric indices
    df = preprocess_location(df, ZIP_CODES_CSV)

    # Normalize age using standard scaling
    age_scaler = joblib.load(AGE_SCALER)
    df["Age"] = age_scaler.transform(df[["Age"]])

    # Tokenize and pad the provider notes
    df = tokenize_text(df, "Provider Notes", 16, NOTES_TOKENIZER)

    # Organize the processed data into a format suitable for model input
    prepared_input = {
        "location_input": df["Encoded Location"].values,
        "notes_input": np.stack(df["Tokenized Provider Notes"].values),
        "other_features_input": df.drop(
            ["Encoded Location", "Tokenized Provider Notes"], axis=1
        ).values,
    }

    return prepared_input


def preprocess_sex(df):
    # Load the fitted OneHotEncoder
    sex_encoder = joblib.load(SEX_ENCODER)

    # Use the loaded encoder to transform the new data
    encoded_input = sex_encoder.transform(df[["Sex at Birth"]])

    # Get the categories used for encoding
    categories = sex_encoder.categories_

    # Generate feature names manually
    feature_names = [f"{df.columns[0]}_{category}" for category in categories[0]]

    # Create dataframe from encoded input
    encoded_sex_df = pd.DataFrame(encoded_input, columns=feature_names)

    # Concatenate the new DataFrame with the original (minus the 'Sex at Birth' column)
    df = pd.concat([df.drop("Sex at Birth", axis=1), encoded_sex_df], axis=1)

    return df


def preprocess_symptoms(df):
    """
    Preprocesses the 'Symptoms' column in the given DataFrame for model training.

    Steps:
    1. Splits the 'Symptoms' strings into lists.
    2. Normalizes the symptoms by stripping whitespace and handling empty or null entries.
    3. Encodes the symptoms using MultiLabelBinarizer for model training.

    Parameters:
    - df: pandas DataFrame containing a 'Symptoms' column.

    Returns:
    - A new DataFrame with the original data (minus the 'Symptoms' column) concatenated with the encoded symptoms.
    """

    # Split 'Symptoms' strings into lists and ensure non-list entries are converted to empty lists
    df["Symptoms"] = (
        df["Symptoms"].str.split(", ").apply(lambda x: x if isinstance(x, list) else [])
    )

    # Normalize symptoms
    def normalize_symptoms(symptoms):
        symptoms_list = [symptom.strip() for symptom in symptoms if symptom.strip()]
        if not symptoms_list or symptoms_list == ["None"]:
            return ["No Symptoms"]
        return symptoms_list

    df["Symptoms"] = df["Symptoms"].apply(normalize_symptoms)

    # Load the fitted MultiLabelBinarizer
    symtpoms_mlb = joblib.load(SYMPTOM_ENCODER)

    # Transform the new data
    symptoms_encoded = symtpoms_mlb.transform(df["Symptoms"])

    # Create a new DataFrame from the encoded data
    symptoms_df = pd.DataFrame(symptoms_encoded, columns=symtpoms_mlb.classes_)

    # Concatenate the new DataFrame with the original (minus the 'Symptoms' column)
    processed_df = pd.concat([df.drop("Symptoms", axis=1), symptoms_df], axis=1)

    return processed_df


def preprocess_location(df, zip_codes_csv_url):
    """
    Merges the given DataFrame with a DataFrame of US ZIP codes to add 'CITY' and 'STATE' information,
    then encodes the 'Location' as unique integer indices.

    Steps:

    1. Load an external CSV file containing US ZIP codes into a DataFrame.
    2. Merge this DataFrame with another DataFrame df on ZIP code columns.
    3. Clean up by dropping unnecessary columns and NaN values.
    4. Combine "CITY" and "STATE" into a new "Location" feature.
    5. Encode the "Location" feature into unique integer indices.
    6. Drop the original "CITY", "STATE", and "Location" columns, leaving the DataFrame ready for further processing or model training.

    Parameters:
    - df: pandas DataFrame containing a 'Zip Code' column.
    - zip_codes_csv_url: String URL to a CSV file containing 'POSTAL_CODE', 'CITY', and 'STATE' columns.

    Returns:
    - The modified DataFrame with an added 'Encoded Location' column and unnecessary columns dropped.
    """

    # Load the external CSV file into a DataFrame
    us_zip_codes_df = pd.read_csv(zip_codes_csv_url, dtype={"POSTAL_CODE": str})

    # Merge the DataFrames on the ZIP code columns
    merged_df = pd.merge(
        df,
        us_zip_codes_df[["POSTAL_CODE", "CITY", "STATE"]],
        left_on="Zip Code",
        right_on="POSTAL_CODE",
        how="left",
    )

    # Drop both 'Zip Code' and 'POSTAL_CODE' columns from the merged DataFrame
    merged_df.drop(["Zip Code", "POSTAL_CODE"], axis=1, inplace=True)

    # Dropping rows where 'CITY' is NaN (no matching ZIP codes in merge)
    merged_df.dropna(subset=["CITY"], inplace=True)

    # Combine "CITY" and "STATE" into a new feature
    # merged_df["Location"] = merged_df["CITY"] + ", " + merged_df["STATE"]
    merged_df["Location"] = merged_df["STATE"]

    # Encode "Location" into unique integer indices
    location_encoder = joblib.load(LOCATION_ENCODER)
    merged_df["Encoded Location"] = location_encoder.transform(merged_df["Location"])

    # Drop the original columns
    merged_df.drop(columns=["CITY", "STATE", "Location"], inplace=True)

    return merged_df


def tokenize_text(df, text_column, max_length, tokenizer_path):
    """
    Preprocesses text data for model inference using a saved tokenizer.

    Parameters:
    - df: DataFrame containing the text data in the specified column.
    - text_column: Name of the column containing text data to be tokenized.
    - max_length: Maximum length of the sequences after padding.
    - tokenizer_path: File path to load the trained tokenizer from.

    Steps:
    1. Load the pre-trained tokenizer from the specified JSON file.
    2. Use the loaded tokenizer to convert text in the given column to sequences of integers.
    3. Pad the sequences to ensure they all have the same length, specified by `max_length`.
    4. Update the DataFrame to include a new column with the padded sequences.
    5. Drop the original text column from the DataFrame.

    Returns:
    - The modified DataFrame with tokenized and padded sequences.
    """

    # Load the tokenizer from the JSON file
    with open(tokenizer_path) as f:
        tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_data)

    # Convert text to sequences using the loaded tokenizer and pad them
    sequences = tokenizer.texts_to_sequences(df[text_column])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

    # Update the DataFrame
    df["Tokenized " + text_column] = list(padded_sequences)
    df.drop(columns=[text_column], inplace=True)

    return df

