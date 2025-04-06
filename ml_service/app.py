from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pickle
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import logging
logging.basicConfig(level=logging.DEBUG)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model_and_vectorizer():
    mlflow.set_tracking_uri("https://dagshub.com/NehmeElio/spam-detector.mlflow")

    artifact_path_vectorizer = "vectorizer.pkl"

    uri_model = f"runs:/4e04fa7c9ad94d5e92f412ed2838fab8/model"
    artifact_uri_vectorizer = f"runs:/4e04fa7c9ad94d5e92f412ed2838fab8/{artifact_path_vectorizer}"

    #artifact_local_path_model = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri_model, dst_path="ml_service/tmp/model")
    artifact_local_path_vectorizer = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri_vectorizer, dst_path="ml_service/tmp/vectorizer")

    custom_path_vectorizer = f"ml_service/tmp/vectorizer/vectorizer.pkl"

    model=mlflow.sklearn.load_model(model_uri=uri_model)
    with open(custom_path_vectorizer, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

class TextData(BaseModel):
    text: str

@app.post("/predict")
def predict_spam_or_ham(data: TextData) -> Dict[str, str]:
    text_transformed = vectorizer.transform([data.text])
    prediction = model.predict(text_transformed)
    result = 'spam' if prediction[0] == 1 else 'ham'
    
    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
