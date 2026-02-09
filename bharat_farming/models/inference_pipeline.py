import pandas as pd
import joblib
from utils_gcs import load_csv_gcs
from google.cloud import storage
import tempfile

BUCKET = "group_project2"
MODEL_PATH = "pipeline_artifacts/final_model.pkl"

def load_model_from_gcs():
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(MODEL_PATH)

    with tempfile.NamedTemporaryFile() as tmp:
        blob.download_to_filename(tmp.name)
        model = joblib.load(tmp.name)

    return model

def run_inference(input_df):
    model = load_model_from_gcs()
    preds = model.predict(input_df)
    return preds