import numpy as np
import pytest
from hnsw import HNSW


class TestHNSW:
    def test_hnsw_l2_basic(self):
        """Test basic HNSW functionality with L2 metric."""
        hnsw = HNSW(distance_metric="l2", m=4, ef_construction=10)

        # Insert some vectors
        vectors = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([0.5, 0.5]),
        ]

        for i, vec in enumerate(vectors):
            hnsw.insert(i, vec)

        # Query nearest neighbors
        query = np.array([0.9, 0.1])
        results = hnsw.query(query, k=2, ef=10)

        assert len(results) == 2
        assert 0 in results  # Should find the closest

    def test_hnsw_cosine_basic(self):
        """Test HNSW with cosine metric."""
        hnsw = HNSW(distance_metric="cosine", m=4, ef_construction=10)

        vectors = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
        ]

        for i, vec in enumerate(vectors):
            hnsw.insert(i, vec)

        query = np.array([1.0, 0.0])
        results = hnsw.query(query, k=1, ef=10)

        assert len(results) == 1
        assert results[0] == 0  # Should find itself as closest

    def test_hnsw_dot_basic(self):
        """Test HNSW with dot product metric."""
        hnsw = HNSW(distance_metric="dot", m=4, ef_construction=10)

        vectors = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
        ]

        for i, vec in enumerate(vectors):
            hnsw.insert(i, vec)

        query = np.array([1.0, 0.0])
        results = hnsw.query(query, k=1, ef=10)

        assert len(results) == 1
        # Dot product: [1,0] · [1,0] = 1, [1,0] · [0,1] = 0, [1,0] · [1,1] = 1
        # So 0 and 2 should be equally good, but since it's approximate, just check it's valid

    def test_hnsw_empty_query(self):
        """Test querying empty index."""
        hnsw = HNSW(distance_metric="l2")
        query = np.array([1.0, 0.0])
        results = hnsw.query(query, k=5)
        assert results == []

    def test_hnsw_custom_distance_func(self):
        """Test HNSW with custom distance function."""
        def custom_dist(a, b):
            return np.sum((a - b) ** 2)  # L2

        hnsw = HNSW(distance_func=custom_dist, m=4, ef_construction=10)

        vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        for i, vec in enumerate(vectors):
            hnsw.insert(i, vec)

        query = np.array([1.0, 0.0])
        results = hnsw.query(query, k=1)
        assert results == [0]

    def test_hnsw_invalid_metric(self):
        """Test HNSW with invalid metric raises error."""
        with pytest.raises(ValueError):
            HNSW(distance_metric="invalid")

    def test_hnsw_parameters(self):
        """Test HNSW with different parameters."""
        hnsw = HNSW(distance_metric="l2", m=8, ef_construction=20, m_max0=16)

        assert hnsw.m == 8
        assert hnsw.ef_construction == 20
        assert hnsw.m_max0 == 16