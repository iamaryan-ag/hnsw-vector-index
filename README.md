# Multi-Tenant HNSW Vector Index

A Python implementation of the Hierarchical Navigable Small World (HNSW) algorithm for efficient approximate nearest neighbor (ANN) search in high-dimensional vector spaces. This project includes multi-tenant isolation and a REST API built with FastAPI.

## Features

- **HNSW Algorithm**: Efficient ANN search with hierarchical graph structure.
- **Multi-Tenant Support**: Isolated vector indices per tenant for secure and scalable deployments.
- **REST API**: FastAPI-based server for inserting vectors and performing searches.
- **Benchmarking**: Compare performance against FAISS library.
- **Type Hints**: Full type annotations for better code maintainability.

## Installation

1. Clone the repository:
   ```bash
   git clone "https://github.com/iamaryan-ag/hnsw-vector-index.git"
   cd hnsw-vector-index
   ```

2. Create Virtual Environment:
    ```bash
    python -m venv .venv
    .\\.venv\\Scripts\\activate
    ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the API Server

Start the FastAPI server to expose the HNSW index via REST endpoints:

```bash
python api_server.py
```

The server will run on `http://0.0.0.0:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

### Using the Library Directly

You can use the HNSW classes directly in your Python code:

```python
from hnsw import MultiTenantHNSW
import numpy as np

# Initialize the index
index = MultiTenantHNSW(m=16, ef_construction=40)

# Insert vectors for a tenant
vectors = np.random.random((100, 128)).astype('float32')
for i, vec in enumerate(vectors):
    index.insert("tenant_1", i, vec)

# Query for nearest neighbors
query_vec = np.random.random(128).astype('float32')
results = index.query("tenant_1", query_vec, k=5)
print(results)
```

### Running Benchmarks

To run the benchmark comparing against FAISS:

```bash
python main.py
```

This will output recall and average latency metrics.

## API Endpoints

### POST /insert

Insert a vector into the index for a specific tenant.

**Request Body:**
```json
{
  "tenant_id": "string",
  "node_id": 0,
  "vector": [0.1, 0.2, ...]
}
```

**Response:**
```json
{
  "status": "success",
  "tenant_id": "string",
  "node_id": 0
}
```

### POST /search

Search for the k nearest neighbors in a tenant's index.

**Request Body:**
```json
{
  "tenant_id": "string",
  "vector": [0.1, 0.2, ...],
  "k": 5,
  "ef": 50
}
```

**Response:**
```json
{
  "tenant_id": "string",
  "top_k_ids": [0, 1, 2, 3, 4]
}
```

## Configuration

- `m`: Number of neighbors per node (default: 16)
- `ef_construction`: Candidate list size during index construction (default: 40)
- `ef`: Candidate list size during search (default: 50)

## Benchmark Results

The included benchmark indexes 2000 vectors across two tenants and queries 20 vectors, comparing recall against FAISS's exact search. Typical results show high recall with low latency.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.