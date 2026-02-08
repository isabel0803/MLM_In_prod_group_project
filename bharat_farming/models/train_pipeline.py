import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils_gcs import load_csv_gcs, save_pickle_to_gcs
from model import get_model
from features import build_preprocessor

BUCKET = "your-bucket-name"   # <— CHANGE THIS
TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
MODEL_OUTPUT_PATH = "artifacts/models/final_model.pkl"

def main():

    print("Loading data from GCS...")
    train = load_csv_gcs(BUCKET, TRAIN_PATH)
    val = load_csv_gcs(BUCKET, VAL_PATH)

    X_train = train.drop("Yield", axis=1)
    y_train = train["Yield"]

    X_val = val.drop("Yield", axis=1)
    y_val = val["Yield"]

    print("Building preprocessor...")
    preprocessor, _, _ = build_preprocessor(X_train)

    print("Initializing model...")
    model = get_model()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred_train = pipeline.predict(X_train)
    y_pred_val = pipeline.predict(X_val)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)

    print(f"TRAIN MAE: {train_mae}")
    print(f"VAL MAE:   {val_mae}")

    print("Saving model to GCS...")
    save_pickle_to_gcs(pipeline, BUCKET, MODEL_OUTPUT_PATH)

    print("Training complete!")


if __name__ == "__main__":
    main()