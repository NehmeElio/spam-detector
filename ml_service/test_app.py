# ml_service/test_app.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_endpoint():
    """Test the prediction endpoint works"""
    response = client.post("/predict", json={"text": "Hello world"})
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result