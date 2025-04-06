from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pickle
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import logging
from prometheus_fastapi_instrumentator import Instrumentator
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app = FastAPI()

# Add Prometheus middleware BEFORE app starts
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


mlflow.set_tracking_uri("https://dagshub.com/NehmeElio/spam-detector.mlflow")

def load_model_and_vectorizer():
    artifact_path_vectorizer = "vectorizer.pkl"
    uri_model = f"runs:/4e04fa7c9ad94d5e92f412ed2838fab8/model"
    artifact_uri_vectorizer = f"runs:/4e04fa7c9ad94d5e92f412ed2838fab8/{artifact_path_vectorizer}"

    artifact_local_path_vectorizer = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri_vectorizer, dst_path="ml_service/tmp/vectorizer")
    custom_path_vectorizer = f"ml_service/tmp/vectorizer/vectorizer.pkl"


    logger.info("Loading model and vectorizer...")
    model = mlflow.sklearn.load_model(model_uri=uri_model)
    with open(custom_path_vectorizer, 'rb') as f:
        vectorizer = pickle.load(f)
    
    logger.info("Model and vectorizer loaded successfully.")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

class TextData(BaseModel):
    text: str



@app.post("/predict")
def predict_spam_or_ham(data: TextData) -> Dict[str, str]:
    start_time = time.time()

    # Log the incoming request
    logger.info(f"Received request: {data.text}")

    text_transformed = vectorizer.transform([data.text])
    prediction = model.predict(text_transformed)
    result = 'spam' if prediction[0] == 1 else 'ham'

    logger.info(f"Prediction: {result}")

    prediction_time = time.time() - start_time
    mlflow.log_metric("prediction_time", prediction_time) 
    mlflow.log_metric("prediction_count", 1) 

    # Return the prediction result
    return {"prediction": result}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
