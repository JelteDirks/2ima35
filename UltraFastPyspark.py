import math
import csv
import random
from typing import Dict, List
import pyspark
import scipy.spatial
import sys

from util import describe_type, sample_edges

from argparse import ArgumentParser
from datetime import datetime
from sklearn.datasets import make_circles, make_moons, make_blobs, make_swiss_roll, make_s_curve
from pyspark import SparkConf, SparkContext

from Plotter import *
from DataReader import *


def get_clustering_data(n_samples=100):
    """
    Retrieves all toy datasets from sklearn
    :return: circles, moons, blobs datasets.
    """
    noisy_circles = make_circles(n_samples=n_samples, factor=.5,
                                 noise=0.05)
    noisy_moons = make_moons(n_samples=n_samples, noise=0.05)
    blobs = make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = make_blobs(n_samples=n_samples,
                        cluster_std=[1.0, 2.5, 0.5],
                        random_state=random_state)

    plt.figure(figsize=(9 * 2 + 3, 13))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                        hspace=.01)

    swiss_roll = make_swiss_roll(n_samples, noise=0.05)

    s_shape = make_s_curve(n_samples, noise=0.05)

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2,
                         'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,
                  'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'eps': .15, 'n_neighbors': 2,
                 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {}),
        (no_structure, {}),
        (swiss_roll, {}),
        (s_shape, {})]

    return datasets

def create_distance_matrix(dataset):
    """
    Creates the distance matrix for a dataset with only vertices. Also adds the edges to a dict.
    :param dataset: dataset without edges
    :return: distance matrix, a dict of all edges and the total number of edges
    """
    vertices = []
    size = 0
    three_d = False
    for line in dataset:
        if len(line) == 2:
            vertices.append([line[0], line[1]])
        elif len(line) == 3:
            vertices.append([line[0], line[1], line[2]])
            three_d = True
    if three_d:
        dict = {}
        for i in range(len(dataset)):
            dict2 = {}
            for j in range(i + 1, len(dataset)):
                dict2[j] = np.sqrt(np.sum(np.square(dataset[i] - dataset[j])))
                size += 1
            dict[i] = dict2

    else:
        d_matrix = scipy.spatial.distance_matrix(vertices, vertices, threshold=1000000)
        dict = {}
        # Run with less edges
        for i in range(len(d_matrix)):
            dict2 = {}
            for j in range(i, len(d_matrix)):
                if i != j:
                    size += 1
                    dict2[j] = d_matrix[i][j]
            dict[i] = dict2
    return dict, size, vertices


def get_key(item):
    """
    returns the sorting criteria for the edges. All edges are sorted from small to large values
    :param item: one item
    :return: returns the weight of the edge
    """
    return item[2]


def local_mst_with_union_find(edges_iter):
    edges = list(edges_iter)
    if not edges:
        return iter([])
    # Sort edges by weight (Kruskal's algorithm requirement)
    edges.sort(key=lambda e: e[2])
    # Union-Find data structure
    parent = {}  # parent[v] = parent of vertex v
    rank = {}    # rank[v] = approximate depth of tree rooted at v

    def find(v):
        """
        Find the root of the set containing v.
        Uses path compression: makes all nodes point directly to root.
        """
        if v not in parent:
            parent[v] = v
            rank[v] = 0
        if parent[v] != v:
            parent[v] = find(parent[v])  # Path compression
        return parent[v]

    def union(u, v):
        """
        Merge the sets containing u and v.
        Uses union by rank: attach smaller tree under larger tree.
        Returns True if sets were different (edge added to MST).
        """
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            return False  # Already in same set, would create cycle
        # Union by rank: attach smaller tree under larger
        if rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
        elif rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        else:
            parent[root_v] = root_u
            rank[root_u] += 1
        return True
    mst = []
    for edge in edges:
        u, v, w = edge
        if union(u, v):
            mst.append(edge)
    return iter(mst)


def create_mst(V: List[int], E: Dict[int, Dict[int, float]], epsilon: float, 
               vertex_coordinates, sc: SparkContext, plot_intermediate=False, plotter=None):
    print("Creating MST (Optimized)")
    n = len(V)
    total_runs = 0

    # Convert adjacency dict to edge list
    edge_list = [(i, j, w) for i, edgeDict in E.items() for j, w in edgeDict.items()]
    m = len(edge_list)
    y = np.power(n, 1 + epsilon)
    x = int(np.ceil(m / y))
    np.random.shuffle(edge_list)
    print(f"Initial: m={m} x={x} y={y} n={n}")

    # Parallelize edges across x partitions
    edges_rdd = sc.parallelize(edge_list, numSlices=x)

    while True:
        total_runs += 1

        # Each partition computes its local MST using Union-Find
        edges_rdd = edges_rdd.mapPartitions(local_mst_with_union_find)

        # Cache since we'll aggregate over this RDD
        edges_rdd.cache()

        # OPTIMIZATION: Single-pass aggregation
        # Collects both edge count (m) and unique vertices (n) in one pass
        # Avoids expensive distinct() shuffle operation
        def seq_op(acc, edge):
            """Sequential operation: process one edge within a partition"""
            edge_count, vertices = acc
            return (edge_count + 1, vertices | {edge[0], edge[1]})

        def comb_op(acc1, acc2):
            """Combiner operation: merge results from different partitions"""
            return (acc1[0] + acc2[0], acc1[1] | acc2[1])

        # Single action to get both metrics
        m, vertex_set = edges_rdd.aggregate(
            (0, set()),  # Initial value: (count=0, empty set)
            seq_op,      # Process edges in each partition
            comb_op      # Combine partition results
        )

        n = len(vertex_set)
        y = int(n ** (1 + epsilon))
        x = math.ceil(m / y)
        print(f"Iteration {total_runs}: m={m} n={n} x={x} y={y}")

        # Convergence check: stop when m ≤ n^(1+ε)
        if m <= y:
            break

        # Repartition for next iteration
        # coalesce with shuffle=True allows increasing partitions
        edges_rdd = edges_rdd.coalesce(numPartitions=x, shuffle=True)
        edges_rdd.cache()

    # Collect final edges (small enough now for sequential processing)
    mst_edges = edges_rdd.collect()
    edges_rdd.unpersist()  # Clean up cached RDD

    # Final MST computation on reduced graph (sequential)
    # Use Union-Find directly for efficiency
    mst_edges.sort(key=lambda e: e[2])

    parent = {}
    rank = {}

    def find(v):
        if v not in parent:
            parent[v] = v
            rank[v] = 0
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            return False
        if rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
        elif rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        else:
            parent[root_v] = root_u
            rank[root_u] += 1
        return True

    mst = []
    for edge in mst_edges:
        u, v, _ = edge
        if union(u, v):
            mst.append(edge)

    print(f"Total iterations: {total_runs}")
    print(f"Final MST has {len(mst)} edges spanning {len(vertex_set)} vertices")
    return mst


def main():
    """
    For every dataset, it creates the mst and plots the clustering
    """
    parser = ArgumentParser()
    parser.add_argument('--test', help='Used for smaller dataset and testing', action='store_true')
    parser.add_argument('--epsilon', help='epsilon [default=1/8]', type=float, default=1 / 8)
    parser.add_argument('--machines', help='Number of machines [default=1]', type=int, default=1)
    args = parser.parse_args()

    a = 1
    n = 2000
    epsilon = 1/8
    c = 1/2
    m = int(a * n**(1+c)) # semi-dense graph
    m = int(n * (n-1) / 2) # complete graph

    print('Start generating MST')
    if args.test:
        print('Test argument given')

    start_time = datetime.now()
    print('Starting time:', start_time)

    datasets = get_clustering_data(n_samples=n)
    names_datasets = ['TwoCircles', 'TwoMoons', 'Varied', 'Aniso', 'Blobs', 'Random', 'swissroll', 'sshape']

    conf = SparkConf().setAppName('MST_EDGE_SAMPLING')
    conf = conf.setMaster("local[12]") # [number] is the amount of cores
    sc = SparkContext.getOrCreate(conf=conf)

    num_clusters = [2, 2, 3, 3, 3, 2, 2, 2]
    cnt = 0
    time = []
    file_location = 'new_results/'
    plotter = Plotter(None, None, file_location)
    data_reader = DataReader()

    for dataset in datasets:
        if cnt >= 3:
            break
        if cnt <= 0:
            cnt += 1
            continue
        print("=========")
        timestamp = datetime.now()
        edges, size, vertex_coordinates = create_distance_matrix(dataset[0][0])
        E, m = sample_edges(E=edges, m=m, seed=1)
        print(f"Reduced |E|={size} to m={m} edges")

        plotter.set_vertex_coordinates(vertex_coordinates)
        plotter.set_dataset(names_datasets[cnt])
        plotter.update_string()
        plotter.reset_round()

        V = list(range(len(vertex_coordinates)))
        timestamp = datetime.now()
        mst = create_mst(V, E, epsilon=epsilon, vertex_coordinates=vertex_coordinates,
                         sc=sc, plot_intermediate=False, plotter=None)
        endtime = datetime.now()
        print('Found MST in: ', endtime - timestamp)
        time.append(datetime.now() - timestamp)
        timestamp = datetime.now()

        if len(vertex_coordinates[0]) > 2:
            plotter.plot_mst_3d(mst, intermediate=False, plot_cluster=False, num_clusters=num_clusters[cnt])
        else:
            plotter.plot_mst_2d(mst, intermediate=False, plot_cluster=False, num_clusters=num_clusters[cnt])
        cnt += 1
        print("=========")

    sc.stop()
    return


if __name__ == '__main__':
    # Initial call to main function
    main()
