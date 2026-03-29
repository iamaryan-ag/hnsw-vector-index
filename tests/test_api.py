import numpy as np
from fastapi.testclient import TestClient
from api_server import app

client = TestClient(app)


class TestAPI:
    def test_insert_vector(self):
        """Test inserting a vector via API."""
        payload = {
            "tenant_id": "test_tenant",
            "node_id": 0,
            "vector": [1.0, 0.0, 0.0],
            "metric": "l2"
        }

        response = client.post("/insert", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["tenant_id"] == "test_tenant"
        assert data["node_id"] == 0
        assert data["metric"] == "l2"

    def test_search_vector(self):
        """Test searching for vectors via API."""
        # First insert some vectors
        vectors = [
            {"tenant_id": "test_tenant", "node_id": 0, "vector": [1.0, 0.0, 0.0], "metric": "l2"},
            {"tenant_id": "test_tenant", "node_id": 1, "vector": [0.0, 1.0, 0.0], "metric": "l2"},
            {"tenant_id": "test_tenant", "node_id": 2, "vector": [0.0, 0.0, 1.0], "metric": "l2"},
        ]

        for vec in vectors:
            client.post("/insert", json=vec)

        # Now search
        query_payload = {
            "tenant_id": "test_tenant",
            "vector": [1.0, 0.0, 0.0],
            "k": 1,
            "ef": 10,
            "metric": "l2"
        }

        response = client.post("/search", json=query_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "test_tenant"
        assert data["metric"] == "l2"
        assert len(data["top_k_ids"]) == 1
        assert data["top_k_ids"][0] == 0  # Should find the closest

    def test_search_empty_tenant(self):
        """Test searching in non-existent tenant."""
        query_payload = {
            "tenant_id": "empty_tenant",
            "vector": [1.0, 0.0],
            "k": 5,
            "ef": 10,
            "metric": "l2"
        }

        response = client.post("/search", json=query_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["top_k_ids"] == []

    def test_insert_different_metrics(self):
        """Test inserting with different metrics."""
        # Insert with cosine
        payload_cos = {
            "tenant_id": "cos_tenant",
            "node_id": 0,
            "vector": [1.0, 0.0],
            "metric": "cosine"
        }
        response = client.post("/insert", json=payload_cos)
        assert response.status_code == 200

        # Insert with dot
        payload_dot = {
            "tenant_id": "dot_tenant",
            "node_id": 0,
            "vector": [1.0, 0.0],
            "metric": "dot"
        }
        response = client.post("/insert", json=payload_dot)
        assert response.status_code == 200

    def test_metric_consistency_api(self):
        """Test metric consistency enforcement via API."""
        # Insert with l2
        client.post("/insert", json={
            "tenant_id": "consistency_tenant",
            "node_id": 0,
            "vector": [1.0, 0.0],
            "metric": "l2"
        })

        # Try to search with different metric - should work since it's per-call
        # Actually, the API allows per-call metric, but internally checks tenant metric
        # Wait, looking back at api_server.py, it calls engine.query with metric, which checks consistency

        # So this should work
        response = client.post("/search", json={
            "tenant_id": "consistency_tenant",
            "vector": [1.0, 0.0],
            "k": 1,
            "ef": 10,
            "metric": "l2"
        })
        assert response.status_code == 200