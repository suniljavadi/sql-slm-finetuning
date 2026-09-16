from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_generate_valid_request():
    payload = {"prompt": "return top 3 products by sales"}
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    assert "sql" in response.json()


def test_generate_invalid_request():
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422
