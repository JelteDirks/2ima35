import math
from pyspark import SparkContext
from typing import List, Tuple, Set, Dict


def reduce_edges(vertices: List[int],
                 E: Dict[int, Dict[int, float]],
                 c: float,
                 epsilon: float,
                 sc: SparkContext = None) -> Tuple[List, Set]:
    """
    Reduces edges by computing MSTs on random vertex partition intersections.
    
    THEORY: This implements the Karger-Klein-Tarjan MST algorithm's edge reduction step.
    By randomly partitioning vertices into k subsets twice (U and V), we create k² 
    subgraphs from their intersections. Computing MSTs on these subgraphs and keeping 
    only MST edges reduces the total edge count while preserving the global MST with 
    high probability.
    
    The key insight: if an edge is not in ANY local MST, it cannot be in the global MST
    (assuming the subgraphs cover all vertex pairs with sufficient probability).
    
    :param vertices: List of vertex IDs (integers)
    :param E: Edge dictionary where E[u][v] = weight (with u < v by convention)
    :param c: Constant parameter for partition size calculation
    :param epsilon: Small positive value for partition size tuning
    :param sc: Optional existing SparkContext (recommended for production)
    :return: Tuple of (mst_edges_list, removed_edges_set)
             mst_edges_list: List of (u, v, weight) tuples in MST
             removed_edges_set: Set of (u, v, weight) tuples removed
    """
    # Input validation
    if not vertices or not E:
        return [], set()

    # Manage SparkContext lifecycle
    context_created = False
    if sc is None:
        sc = SparkContext.getOrCreate()
        context_created = True

    try:
        n = len(vertices)
        k = math.ceil(n**((c - epsilon) / 2))
        print(f"Vertices: {n}, Partitions: {k}, Subgraphs: {k**2}")

        # Partition vertices
        U, V = partition_vertices(vertices, k)

        # IMPROVEMENT 1: Broadcast edges for efficient access
        # THEORY: The edge dictionary E can be large. Broadcasting it ensures
        # each worker receives it once and caches it, rather than serializing
        # it with every task. For a graph with millions of edges, this saves
        # significant network bandwidth.
        E_broadcast = sc.broadcast(E)

        # IMPROVEMENT 2: Create index pairs as RDD directly
        # THEORY: Instead of creating k² partition objects in memory and using
        # cartesian (which materializes all combinations), we generate lightweight
        # index pairs. This reduces memory from O(k² × partition_size) to O(k²).
        partition_pairs = [(i, j) for i in range(k) for j in range(k)]

        # PARALLELISM TUNING: Use 4× cores to handle load imbalance
        # THEORY: Some subgraphs may be denser than others. Over-partitioning
        # allows Spark's scheduler to balance work dynamically across workers.
        num_slices = min(len(partition_pairs), sc.defaultParallelism * 4)
        rdd_pairs = sc.parallelize(partition_pairs, numSlices=num_slices)

        # IMPROVEMENT 3: Process subgraphs in parallel without collecting intermediate results
        def process_subgraph(
                pair_idx: Tuple[int, int]) -> List[Tuple[int, int, float]]:
            """
            Process a single subgraph (U_i ∩ V_j) and return its MST edges.
            
            THEORY: This function runs on worker nodes. Each worker:
            1. Accesses the broadcasted edge dictionary (O(1) lookup)
            2. Extracts the subgraph induced by U_i ∪ V_j
            3. Computes MST using Kruskal's algorithm
            4. Returns only MST edges (filters out non-MST edges here)
            
            :param pair_idx: Tuple of (i, j) indicating U_i and V_j partition indices
            :return: List of MST edges as (u, v, weight) tuples
            """
            i, j = pair_idx
            u_partition = U[i]
            v_partition = V[j]
            edges_dict = E_broadcast.value

            # Extract subgraph edges
            subgraph_vertices, subgraph_edges = get_edges(
                u_partition, v_partition, edges_dict)

            # Compute MST for this subgraph
            if subgraph_edges:
                mst_edges = find_mst(subgraph_vertices, subgraph_edges)
                return mst_edges
            return []

        # EXPLANATION: flatMap applies process_subgraph to each (i,j) pair
        # flatMap vs map:
        #   - map would give us RDD[List[edges]] (nested lists)
        #   - flatMap gives us RDD[edge] (flat list of all edges)
        # THEORY: This keeps data distributed across the cluster. We avoid
        # collecting k² lists of MST edges to the driver, which could be
        # hundreds of GB for large graphs.
        mst_edges_rdd = rdd_pairs.flatMap(process_subgraph)

        # IMPROVEMENT 4: Use distinct() to remove duplicate edges
        # THEORY: The same edge can appear in multiple subgraphs. For example,
        # if edge (u,v) exists and u∈U₁,U₃ and v∈V₂,V₄, then (u,v) appears
        # in subgraphs (U₁,V₂), (U₁,V₄), (U₃,V₂), (U₃,V₄).
        #
        # distinct() implementation:
        # 1. Hashes each edge
        # 2. Shuffles edges with same hash to same partition
        # 3. Within each partition, uses a HashSet to eliminate duplicates
        # Time: O(E) with one shuffle operation
        unique_mst_edges = mst_edges_rdd.distinct()

        # Collect final results (acceptable since MST has only O(n) edges)
        mst = unique_mst_edges.collect()

        # IMPROVEMENT 5: Compute removed edges efficiently using set operations
        # THEORY: Set difference is O(|E|) and happens on driver (acceptable
        # since we need the final result here anyway). We convert both edge
        # representations to a normalized tuple format for comparison.
        mst_set = set(mst)

        # Convert dictionary to set of tuples for comparison
        all_edges_set = set()
        for u in E:
            for v in E[u]:
                all_edges_set.add((u, v, E[u][v]))

        removed_edges = all_edges_set - mst_set

        # Cleanup broadcast variable to free memory on workers
        E_broadcast.unpersist()

        return mst, removed_edges

    finally:
        # Only stop context if we created it
        if context_created:
            sc.stop()


def partition_vertices(vertices: List[int],
                       k: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Randomly partition vertices into k subsets twice.
    
    THEORY: Random partitioning is crucial for the algorithm's correctness.
    
    Coverage Analysis:
    - For an edge (u,v), it appears in subgraph (Uᵢ, Vⱼ) iff u∈Uᵢ AND v∈Vⱼ
    - Probability u∈Uᵢ = 1/k (uniform random assignment)
    - Probability v∈Vⱼ = 1/k (independent random assignment)
    - Probability edge appears in a specific subgraph = 1/k²
    - Expected number of subgraphs containing edge = k² × (1/k²) = 1
    
    Using TWO independent partitions U and V (not just U and U) is essential:
    - With U×U: edges within same partition appear in only 1 subgraph
    - With U×V: edges distributed more uniformly across subgraphs
    
    :param vertices: List of all vertex IDs
    :param k: Number of partitions
    :return: Tuple of (U_partitions, V_partitions) where each is a list of k lists
    """
    import random

    def create_partitions(verts: List[int],
                          num_partitions: int) -> List[List[int]]:
        """
        Create k random partitions of vertices.
        
        ALGORITHM:
        1. Shuffle vertices (Fisher-Yates shuffle: O(n) time, uniform random)
        2. Distribute round-robin into k partitions
        
        PROPERTY: Each partition gets ⌊n/k⌋ or ⌈n/k⌉ vertices (balanced)
        """
        shuffled = verts.copy()
        random.shuffle(shuffled)

        partitions = [[] for _ in range(num_partitions)]
        for idx, v in enumerate(shuffled):
            partitions[idx % num_partitions].append(v)

        return partitions

    U = create_partitions(vertices, k)
    V = create_partitions(vertices, k)

    return U, V


def get_edges(
    u_partition: List[int], v_partition: List[int], edges: Dict[int,
                                                                Dict[int,
                                                                     float]]
) -> Tuple[Set[int], List[Tuple[int, int, float]]]:
    """
    Extract subgraph edges between two vertex partitions.
    
    THEORY: A subgraph is defined by:
    - Vertex set: Uᵢ ∪ Vⱼ
    - Edge set: {(u,v,w) ∈ E | u,v ∈ Uᵢ ∪ Vⱼ}
    
    This is the induced subgraph on the vertex set Uᵢ ∪ Vⱼ.
    
    IMPLEMENTATION NOTE: The edge dictionary stores edges as E[min(u,v)][max(u,v)] = w
    This is a space-efficient representation that avoids storing both (u,v) and (v,u).
    
    :param u_partition: Vertices in partition Uᵢ
    :param v_partition: Vertices in partition Vⱼ
    :param edges: Edge dictionary E[u][v] = weight where u < v
    :return: Tuple of (subgraph_vertices, subgraph_edges)
             subgraph_vertices: Set of vertex IDs in this subgraph
             subgraph_edges: List of (u, v, weight) tuples
    """
    # STEP 1: Combine partitions into subgraph vertex set
    # THEORY: Set union is O(|Uᵢ| + |Vⱼ|) ≈ O(n/k)
    subgraph_vertices = set(u_partition) | set(v_partition)

    # STEP 2: Extract edges with both endpoints in subgraph
    # ALGORITHM: For each edge (u,v,w) in E, check if u,v ∈ subgraph_vertices
    # OPTIMIZATION: Only iterate over edges where u ∈ subgraph_vertices
    # This reduces checks from O(|E|) to O(|subgraph_vertices| × avg_degree)
    subgraph_edges = []

    for u in subgraph_vertices:
        if u in edges:
            for v, weight in edges[u].items():
                # Since edges[u][v] means u < v, we only need to check if v is in subgraph
                if v in subgraph_vertices:
                    subgraph_edges.append((u, v, weight))

    return subgraph_vertices, subgraph_edges


def find_mst(
        vertices: Set[int],
        edges: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    """
    Find MST using Kruskal's algorithm with Union-Find.
    
    THEORY - Kruskal's Algorithm:
    1. Sort edges by weight (ascending)
    2. Initialize each vertex as its own connected component
    3. For each edge (u,v,w) in sorted order:
       - If u and v are in different components, add edge to MST and merge components
       - Otherwise, skip edge (would create a cycle)
    
    CORRECTNESS (Greedy Exchange Argument):
    - At each step, we add the minimum-weight edge that doesn't create a cycle
    - This is optimal because any MST must include such edges (cut property)
    - The algorithm terminates with exactly |V|-1 edges (tree property)
    
    TIME COMPLEXITY:
    - Sorting: O(E log E)
    - Union-Find operations: O(E × α(V)) where α is inverse Ackermann
    - Total: O(E log E) since α(V) < 5 for all practical V
    
    SPACE COMPLEXITY: O(V) for Union-Find structure
    
    :param vertices: Set of vertex IDs in subgraph
    :param edges: List of (u, v, weight) tuples
    :return: List of MST edges as (u, v, weight) tuples
    """
    if not edges:
        return []

    # STEP 1: Sort edges by weight
    # THEORY: We must consider edges in non-decreasing weight order to ensure
    # we always add the minimum-weight edge that doesn't create a cycle
    sorted_edges = sorted(edges, key=lambda e: e[2])

    # STEP 2: Initialize Union-Find data structure
    # THEORY: Union-Find maintains disjoint sets (connected components)
    # Operations:
    #   - find(v): returns representative of component containing v
    #   - union(u,v): merges components containing u and v

    parent = {
        v: v
        for v in vertices
    }  # Initially, each vertex is its own parent
    rank = {v: 0 for v in vertices}  # Rank used for union-by-rank optimization

    def find(v: int) -> int:
        """
        Find the root (representative) of the component containing v.
        
        OPTIMIZATION: Path compression - make all nodes on path point directly to root
        EFFECT: Amortizes future find operations to nearly O(1)
        IMPLEMENTATION: After finding root recursively, update parent[v] = root
        """
        if parent[v] != v:
            parent[v] = find(parent[v])  # Path compression
        return parent[v]

    def union(u: int, v: int) -> bool:
        """
        Merge components containing u and v.
        
        OPTIMIZATION: Union by rank - attach smaller tree under larger tree
        EFFECT: Keeps tree height logarithmic, ensuring fast find operations
        
        :return: True if components were merged (u and v were in different components)
                 False if already in same component (would create cycle)
        """
        root_u, root_v = find(u), find(v)

        if root_u == root_v:
            return False  # Already in same component - would create cycle

        # Union by rank: attach shorter tree under taller tree
        if rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
        elif rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        else:
            parent[root_v] = root_u
            rank[root_u] += 1  # Only increment rank when equal heights merge

        return True

    # STEP 3: Build MST by adding edges that don't create cycles
    mst_edges = []

    for edge in sorted_edges:
        u, v, weight = edge

        if union(u, v):
            mst_edges.append(edge)

            # OPTIMIZATION: Early termination
            # THEORY: An MST on n vertices has exactly n-1 edges
            if len(mst_edges) == len(vertices) - 1:
                break

    return mst_edges
