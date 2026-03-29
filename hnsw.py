import numpy as np
import heapq
from typing import List, Dict, Set, Tuple, Callable


class HNSW:
    """Hierarchical Navigable Small World (HNSW) index for approximate nearest neighbor search."""

    def __init__(self, distance_func: Callable[[np.ndarray, np.ndarray], float], m=16, ef_construction=40, m_max0=32):
        """Initialize the HNSW index.

        Args:
            distance_func: Function to compute distance between two vectors.
            m: Number of neighbors to keep per node in the graph.
            ef_construction: Size of the dynamic candidate list during index construction.
            m_max0: Maximum number of neighbors at layer 0.
        """
        self.distance_func = distance_func
        self.m = m
        self.ef_construction = ef_construction
        self.m_max = m
        self.m_max0 = m_max0
        self.level_mult = 1 / np.log(m)
        self.graphs: List[Dict[int, Set[int]]] = []
        self.data: Dict[int, np.ndarray] = {}
        self.entry_point: int = None
        self.max_level = -1

    def _get_random_level(self) -> int:
        """Generate a random level for a new node based on the level multiplier.

        Returns:
            The random level as an integer.
        """
        return int(np.floor(-np.log(np.random.random()) * self.level_mult))

    def _greedy_search(self, query: np.ndarray, start_node: int, layer: int) -> int:
        """Perform greedy search to find the closest node at a given layer.

        Args:
            query: The query vector.
            start_node: The node to start the search from.
            layer: The layer to search in.

        Returns:
            The ID of the closest node found.
        """
        curr_node = start_node
        curr_dist = self.distance_func(query, self.data[curr_node])
        changed = True
        while changed:
            changed = False
            for neighbor in self.graphs[layer].get(curr_node, []):
                dist = self.distance_func(query, self.data[neighbor])
                if dist < curr_dist:
                    curr_dist = dist
                    curr_node = neighbor
                    changed = True
        return curr_node

    def insert(self, node_id: int, vec: np.ndarray):
        """Insert a new vector into the HNSW index.

        Args:
            node_id: Unique identifier for the node.
            vec: The vector to insert.
        """
        self.data[node_id] = vec
        level = self._get_random_level()
        if self.entry_point is None:
            for l in range(level + 1):
                self.graphs.append({node_id: set()})
            self.entry_point = node_id
            self.max_level = level
            return
        while len(self.graphs) <= max(level, self.max_level):
            self.graphs.append({})
        curr_node = self.entry_point
        for l in range(self.max_level, level, -1):
            curr_node = self._greedy_search(vec, curr_node, l)
        for l in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(vec, curr_node, self.ef_construction, l)
            neighbors = self._select_neighbors(vec, candidates, self.m)
            self.graphs[l][node_id] = set(neighbors)
            for neighbor in neighbors:
                if neighbor not in self.graphs[l]:
                    self.graphs[l][neighbor] = set()
                self.graphs[l][neighbor].add(node_id)
                limit = self.m_max0 if l == 0 else self.m_max
                if len(self.graphs[l][neighbor]) > limit:
                    new_conn = self._select_neighbors(self.data[neighbor], list(self.graphs[l][neighbor]), limit)
                    self.graphs[l][neighbor] = set(new_conn)
            if neighbors:
                curr_node = neighbors[0]
        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

    def _search_layer(self, query: np.ndarray, entry_node: int, ef: int, layer: int) -> List[int]:
        """Search for the ef closest neighbors at a specific layer.

        Args:
            query: The query vector.
            entry_node: The node to start the search from.
            ef: The size of the candidate list.
            layer: The layer to search in.

        Returns:
            A list of the closest node IDs.
        """
        visited = {entry_node}
        dist = self.distance_func(query, self.data[entry_node])
        candidates = [(dist, entry_node)]
        results = [(dist, entry_node)]
        heapq.heapify(candidates)
        while candidates:
            curr_dist, curr_node = heapq.heappop(candidates)
            if curr_dist > results[-1][0] and len(results) >= ef:
                break
            for neighbor in self.graphs[layer].get(curr_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    d = self.distance_func(query, self.data[neighbor])
                    if d < results[-1][0] or len(results) < ef:
                        heapq.heappush(candidates, (d, neighbor))
                        results.append((d, neighbor))
                        results.sort()
                        if len(results) > ef:
                            results.pop()
        return [node for d, node in results]

    def _select_neighbors(self, vec: np.ndarray, candidates: List[int], m: int) -> List[int]:
        """Select the m closest neighbors from a list of candidates.

        Args:
            vec: The reference vector.
            candidates: List of candidate node IDs.
            m: Number of neighbors to select.

        Returns:
            A list of the selected neighbor IDs.
        """
        dists = [(self.distance_func(vec, self.data[c]), c) for c in candidates]
        dists.sort()
        return [c for d, c in dists[:m]]

    def query(self, vec: np.ndarray, k=5, ef=50) -> List[int]:
        """Query the index for the k nearest neighbors.

        Args:
            vec: The query vector.
            k: Number of neighbors to return.
            ef: Size of the candidate list during search.

        Returns:
            A list of the k nearest neighbor IDs.
        """
        if not self.data:
            return []
        curr_node = self.entry_point
        for l in range(self.max_level, 0, -1):
            curr_node = self._greedy_search(vec, curr_node, l)
        candidates = self._search_layer(vec, curr_node, ef, 0)
        return candidates[:k]


class MultiTenantHNSW:
    """Multi-tenant HNSW index that isolates graphs per tenant."""

    def __init__(self, m=16, ef_construction=40):
        """Initialize the multi-tenant HNSW index.

        Args:
            m: Number of neighbors to keep per node in the graph.
            ef_construction: Size of the dynamic candidate list during index construction.
        """
        self.tenants: Dict[str, HNSW] = {}
        self.m = m
        self.ef_construction = ef_construction

    @staticmethod
    def _l2_dist(a: np.ndarray, b: np.ndarray) -> float:
        """Compute L2 (Euclidean) distance between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            The Euclidean distance.
        """
        return np.linalg.norm(a - b)

    def insert(self, tenant_id: str, node_id: int, vec: np.ndarray):
        """Insert a vector for a specific tenant.

        Args:
            tenant_id: The tenant identifier.
            node_id: Unique node ID within the tenant.
            vec: The vector to insert.
        """
        if tenant_id not in self.tenants:
            self.tenants[tenant_id] = HNSW(self._l2_dist, self.m, self.ef_construction)
        self.tenants[tenant_id].insert(node_id, vec)

    def query(self, tenant_id: str, vec: np.ndarray, k=5, ef=50) -> List[int]:
        """Query the index for a specific tenant.

        Args:
            tenant_id: The tenant identifier.
            vec: The query vector.
            k: Number of neighbors to return.
            ef: Size of the candidate list during search.

        Returns:
            A list of the k nearest neighbor IDs for the tenant.
        """
        if tenant_id not in self.tenants:
            return []
            return []
        return self.tenants[tenant_id].query(vec, k, ef)