import numpy as np
import time
import faiss
from hnsw import MultiTenantHNSW


def extended_benchmark():
    """Run an extended benchmark comparing MultiTenantHNSW with FAISS.

    This function indexes vectors into two tenants, then queries and compares
    recall against FAISS's exact search.
    """
    dim, n_points, n_queries = 64, 2000, 20
    data = np.random.random((n_points, dim)).astype('float32')
    queries = np.random.random((n_queries, dim)).astype('float32')

    # Multi-tenant Test
    mt_hnsw = MultiTenantHNSW()
    print("Indexing two separate tenants...")
    for i in range(n_points // 2):
        mt_hnsw.insert("tenant_A", i, data[i])
        mt_hnsw.insert("tenant_B", i + (n_points // 2), data[i + (n_points // 2)])

    # Benchmark vs FAISS
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(data)

    start = time.time()
    hits = 0
    for i in range(n_queries):
        # Query only tenant A's namespace
        res = mt_hnsw.query("tenant_A", queries[i], k=1, ef=100)
        _, f_res = faiss_index.search(queries[i:i+1], 1)
        if res and res[0] == f_res[0][0]:
            hits += 1

    duration = time.time() - start
    print(f"Multi-tenant Search Recall (Tenant A only): {hits/n_queries:.2%}")
    print(f"Average Latency per Query: {duration/n_queries*1000:.2f}ms")


if __name__ == "__main__":
    extended_benchmark()