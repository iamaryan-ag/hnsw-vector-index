import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
from hnsw import MultiTenantHNSW

app = FastAPI(title="Multi-Tenant HNSW Service")
engine = MultiTenantHNSW()

class VectorItem(BaseModel):
    tenant_id: str
    node_id: int
    vector: List[float]

class QueryItem(BaseModel):
    tenant_id: str
    vector: List[float]
    k: int = 5
    ef: int = 50

@app.post("/insert")
async def insert_vector(item: VectorItem):
    """Insert a vector into the multi-tenant HNSW index.

    Args:
        item: The vector item containing tenant_id, node_id, and vector.

    Returns:
        A dict with status, tenant_id, and node_id.
    """
    vec = np.array(item.vector, dtype='float32')
    engine.insert(item.tenant_id, item.node_id, vec)
    return {"status": "success", "tenant_id": item.tenant_id, "node_id": item.node_id}

@app.post("/search")
async def search_vector(item: QueryItem):
    """Search for nearest neighbors in the multi-tenant HNSW index.

    Args:
        item: The query item containing tenant_id, vector, k, and ef.

    Returns:
        A dict with tenant_id and top_k_ids.
    """
    vec = np.array(item.vector, dtype='float32')
    results = engine.query(item.tenant_id, vec, k=item.k, ef=item.ef)
    return {"tenant_id": item.tenant_id, "top_k_ids": [int(r) for r in results]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)