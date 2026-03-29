import numpy as np
import pytest
from benchmark import recall_at_k, build_faiss_index, compute_ground_truth, run_benchmark


class TestBenchmark:
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        # Perfect match
        result_ids = [0, 1, 2]
        gt_ids = [0, 1, 2]
        recall = recall_at_k(result_ids, gt_ids, 3)
        assert recall == 1.0

        # Partial match
        result_ids = [0, 1, 3]
        gt_ids = [0, 1, 2]
        recall = recall_at_k(result_ids, gt_ids, 3)
        assert recall == 2.0 / 3.0

        # No match
        result_ids = [3, 4, 5]
        gt_ids = [0, 1, 2]
        recall = recall_at_k(result_ids, gt_ids, 3)
        assert recall == 0.0

        # Empty ground truth
        recall = recall_at_k([0, 1], [], 2)
        assert recall == 0.0

    def test_build_faiss_index_l2(self):
        """Test FAISS index building with L2."""
        data = np.random.random((10, 4)).astype("float32")
        index = build_faiss_index(data, "l2")
        assert index is not None
        assert index.ntotal == 10

    def test_build_faiss_index_cosine(self):
        """Test FAISS index building with cosine."""
        data = np.random.random((10, 4)).astype("float32")
        index = build_faiss_index(data, "cosine")
        assert index is not None
        assert index.ntotal == 10

    def test_build_faiss_index_dot(self):
        """Test FAISS index building with dot product."""
        data = np.random.random((10, 4)).astype("float32")
        index = build_faiss_index(data, "dot")
        assert index is not None
        assert index.ntotal == 10

    def test_build_faiss_index_invalid_metric(self):
        """Test FAISS index building with invalid metric."""
        data = np.random.random((10, 4)).astype("float32")
        with pytest.raises(ValueError):
            build_faiss_index(data, "invalid")

    def test_compute_ground_truth_l2(self):
        """Test ground truth computation with L2."""
        data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="float32")
        queries = np.array([[1.0, 0.0]], dtype="float32")
        index = build_faiss_index(data, "l2")
        gt = compute_ground_truth(index, queries, 2, "l2")
        assert len(gt) == 1
        assert len(gt[0]) == 2
        assert gt[0][0] == 0  # Closest should be index 0

    def test_compute_ground_truth_cosine(self):
        """Test ground truth computation with cosine."""
        data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        queries = np.array([[1.0, 0.0]], dtype="float32")
        index = build_faiss_index(data, "cosine")
        gt = compute_ground_truth(index, queries, 1, "cosine")
        assert len(gt) == 1
        assert gt[0][0] == 0

    def test_run_benchmark_small(self):
        """Test running a small benchmark."""
        # Use very small parameters for fast testing
        results = run_benchmark(
            dim=4,
            n_points=20,
            n_queries=5,
            k=2,
            m=4,
            ef_construction=10,
            ef_search=20,
            metric="l2"
        )

        # Check that all expected keys are present
        expected_keys = [
            "metric", "m", "ef_construction", "ef_search",
            "hnsw_build_time", "hnsw_qps", "faiss_build_time",
            "recall_at_k", "hnsw_memory_bytes", "faiss_memory_bytes"
        ]

        for key in expected_keys:
            assert key in results

        # Check reasonable value ranges
        assert results["recall_at_k"] >= 0.0
        assert results["recall_at_k"] <= 1.0
        assert results["hnsw_qps"] > 0
        assert results["hnsw_memory_bytes"] > 0
        assert results["faiss_memory_bytes"] > 0