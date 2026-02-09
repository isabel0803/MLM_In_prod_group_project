import gcsfs
import pandas as pd
import joblib
from google.cloud import storage

PROJECT_ID = 'MODIFY_THIS'

# Load CSV from GCS
def load_csv_gcs(bucket: str, path: str):
    fs = gcsfs.GCSFileSystem()
    full_path = f"{bucket}/{path}"
    with fs.open(full_path) as f:
        return pd.read_csv(f)

# Save any python object (model, transformer, etc.) to GCS
def save_pickle_to_gcs(obj, bucket: str, path: str):
    import tempfile

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket)
    blob = bucket.blob(path)

    with tempfile.NamedTemporaryFile() as tmp:
        joblib.dump(obj, tmp.name)
        blob.upload_from_filename(tmp.name)