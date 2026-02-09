import joblib
from google.cloud import storage
import io
from utils_gcs import PROJECT_ID, load_csv_gcs, save_csv_to_gcs

# Configuration
BUCKET_NAME = "group_project2"
MODEL_PATH = "pipeline_artifacts/final_model.pkl"
DATA_PATH = "inputs/new_data.csv"
OUTPUT_PATH = "outputs/predictions.csv"


def load_model_from_gcs():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)

    # Download as bytes and load directly with joblib
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    return joblib.load(buffer)



def run_inference():
    print("Loading model...")
    model = load_model_from_gcs()

    print("Loading data...")
    input_df = load_csv_gcs(path=DATA_PATH, bucket=BUCKET_NAME)

    print("Running inference...")
    preds = model.predict(input_df)

    input_df['predictions'] = preds

    print("Saving results...")
    save_csv_to_gcs(input_df, BUCKET_NAME, OUTPUT_PATH)
    return input_df


if __name__ == "__main__":
    result_df = run_inference()
    print(result_df.head())