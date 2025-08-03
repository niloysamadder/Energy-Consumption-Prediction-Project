from azure.storage.blob import BlobServiceClient
import os

def download_blob_if_not_exists(container_name, blob_name, download_path, connection_string):
    if not os.path.exists(download_path):
        print(f"Downloading {blob_name} from Azure Blob Storage ...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)
        with open(download_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        print(f"✅ {blob_name} downloaded.")

# At app startup (before model loading!):
CONTAINER = "models"
CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

os.makedirs("models", exist_ok=True)
model_files = ["best_rf_model.pkl", "cnn_model.keras", "rnn_model.keras", "scaler.pkl"]

for model_file in model_files:
    download_blob_if_not_exists(CONTAINER, model_file, os.path.join("models", model_file), CONNECTION_STRING)



from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model

app = FastAPI()
ui_path = os.path.join(os.path.dirname(__file__), "ui")
app.mount("/ui", StaticFiles(directory=ui_path), name="ui")

FEATURES = [
    'Energy_Production_MWh',
    'Type_of_Renewable_Energy',
    'Installed_Capacity_MW',
    'Energy_Storage_Capacity_MWh',
    'Storage_Efficiency_Percentage',
    'Grid_Integration_Level'
]

numeric_features = [
    'Energy_Production_MWh',
    'Installed_Capacity_MW',
    'Energy_Storage_Capacity_MWh',
    'Storage_Efficiency_Percentage',
    'Grid_Integration_Level'
]

scaler = joblib.load(os.path.join("..", "..", "models", "scaler.pkl"))
rf_model = joblib.load(os.path.join("..", "..", "models", "best_rf_model.pkl"))
cnn_model = load_model(os.path.join("..", "..", "models", "cnn_model.keras"))
rnn_model = load_model(os.path.join("..", "..", "models", "rnn_model.keras"))

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(ui_path, "index.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict(features: dict, model_type: str = Query("rf")):
    # Convert incoming to DataFrame for scaling
    X_input = np.array([[features.get(feat, 0) for feat in FEATURES]])
    X_df = {feat: [features.get(feat, 0)] for feat in FEATURES}
    import pandas as pd
    X_df = pd.DataFrame(X_df)
    X_df[numeric_features] = scaler.transform(X_df[numeric_features])
    X_scaled = X_df.values

    if model_type == "cnn":
        X_dl = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        y_pred = cnn_model.predict(X_dl)
    elif model_type == "rnn":
        X_dl = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        y_pred = rnn_model.predict(X_dl)
    else:
        y_pred = rf_model.predict(X_scaled)
    return {"prediction": float(y_pred[0][0] if hasattr(y_pred[0], '__len__') else y_pred[0])}
