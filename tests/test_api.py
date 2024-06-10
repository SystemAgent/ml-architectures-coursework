from fastapi.testclient import TestClient
from main import app  # assuming your FastAPI app is saved as main.py

client = TestClient(app)


def test_predict_valid_date():
    response = client.post('/predict/', json={'date': '2023-06-10'})
    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    assert isinstance(data['prediction'], float)


def test_predict_invalid_date():
    response = client.post('/predict/', json={'date': 'invalid-date'})
    assert response.status_code == 400
    assert 'detail' in response.json()


def test_predict_future_date():
    response = client.post('/predict/', json={'date': '2050-01-01'})
    assert response.status_code == 200  # Assuming future dates are handled by the model
    data = response.json()
    assert 'prediction' in data
    assert isinstance(data['prediction'], float)
