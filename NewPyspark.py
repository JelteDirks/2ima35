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


def find_mst(U, V, E):
    """
    finds the mst of graph G = (U union V, E)
    :param U: vertices U
    :param V: vertices V
    :param E: edges of the graph
    :return: the mst and edges not in the mst of the graph
     """
    vertices = set()
    for v in V:
        vertices.add(v)
    for u in U:
        vertices.add(u)
    E = sorted(E, key=get_key)
    connected_component = set()
    mst = []
    remove_edges = set()
    while len(mst) < len(vertices) - 1 and len(connected_component) < len(vertices):
        if len(E) == 0:
            break
        change = False
        i = 0
        while i < len(E):
            if len(connected_component) == 0:
                connected_component.add(E[i][0])
                connected_component.add(E[i][1])
                mst.append(E[i])
                change = True
                E.remove(E[i])
                break
            else:
                if E[i][0] in connected_component:
                    if E[i][1] in connected_component:
                        remove_edges.add(E[i])
                        E.remove(E[i])
                    else:
                        connected_component.add(E[i][1])
                        mst.append(E[i])
                        E.remove(E[i])
                        change = True
                        break
                elif E[i][1] in connected_component:
                    if E[i][0] in connected_component:
                        remove_edges.add(E[i])
                        E.remove(E[i])
                    else:
                        connected_component.add(E[i][0])
                        mst.append(E[i])
                        E.remove(E[i])
                        change = True
                        break
                else:
                    i += 1
        if not change:
            if len(E) != 0:
                connected_component.add(E[0][0])
                connected_component.add(E[0][1])
                mst.append(E[0])
                E.remove(E[0])
    for edge in E:
        remove_edges.add(edge)

    return mst, remove_edges


def local_mst(edges_iter):
    edges = list(edges_iter)

    if not edges:
        return iter([])

    vertices = set()
    for u, v, _ in edges:
        vertices.add(u)
        vertices.add(v)

    mst, _ = find_mst(vertices, vertices, edges)

    return iter([mst])

def create_mst(V:List[int], E:Dict[int, Dict[int, float]], epsilon:float, vertex_coordinates, sc:SparkContext, plot_intermediate=False, plotter=None):
    """
    Creates the mst of the graph G = (V, E).
    As long as the number of edges is greater than n ^(1 + epsilon), the number of edges is reduced
    Then the edges that needs to be removed are removed from E and the size is updated.
    :param plotter: class to plot graphs
    :param V: Vertices
    :param E: edges
    :param epsilon:
    :param plot_intermediate: boolean to indicate if intermediate steps should be plotted
    :param vertex_coordinates: coordinates of vertices
    :return: returns the reduced graph with at most np.power(n, 1 + epsilon) edges
    """
    print("Creating MST")
    n = len(V)
    total_runs = 0
    edge_list = [(i, j, w) for i, edgeDict in E.items() for j, w in edgeDict.items()]
    m = len(edge_list)
    y = np.power(n, 1 + epsilon)
    x = int(np.ceil(m / y))
    np.random.shuffle(edge_list)
    print(f"m={m} x={x} y={y}")

    edges_rdd = sc.parallelize(edge_list, numSlices=x)
    m = edges_rdd.count()
    while True:
        total_runs += 1
        local_results = edges_rdd.mapPartitions(local_mst)
        edges_rdd = local_results.flatMap(lambda mst_edges: mst_edges)

        n = edges_rdd.flatMap(lambda edge: [edge[0], edge[1]]).distinct().count()
        m = edges_rdd.count() 
        y = int(n**(1+epsilon))
        x = math.ceil(m/y)
        print(f"m={m} x={x} y={y}")

        if m <= y:
            break

        edges_rdd = edges_rdd.coalesce(numPartitions=x,shuffle=True)

        continue #skip plotting for now
        if plotter is not None:
            plotter.next_round()

        if plot_intermediate and plotter is not None:
            if len(vertex_coordinates[0]) > 2:
                plotter.plot_mst_3d(mst, intermediate=True, plot_cluster=False, plot_num_machines=1)
            else:
                plotter.plot_mst_2d(mst, intermediate=True, plot_cluster=False, plot_num_machines=1)

    mst_vertices = set(edges_rdd
                    .flatMap(lambda edge: [edge[0], edge[1]])
                    .distinct()
                    .collect())
    mst_edges = edges_rdd.collect()

    mst, _ = find_mst(mst_vertices, mst_vertices, mst_edges)

    print(f"Total runs: {total_runs}")
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

    a = 3
    n = 200
    epsilon = 1/8
    c = 1/2
    m = int(a * n**(1+c)) # semi-dense graph
    #m = int(n * (n-1) / 2) # complete graph

    print('Start generating MST')
    if args.test:
        print('Test argument given')

    start_time = datetime.now()
    print('Starting time:', start_time)

    datasets = get_clustering_data(n_samples=n)
    names_datasets = ['TwoCircles', 'TwoMoons', 'Varied', 'Aniso', 'Blobs', 'Random', 'swissroll', 'sshape']
    # datasets = []

    conf = SparkConf().setAppName('MST_EDGE_SAMPLING')
    conf = conf.setMaster("local[4]") # [number] is the amount of cores
    sc = SparkContext.getOrCreate(conf=conf)

    num_clusters = [2, 2, 3, 3, 3, 2, 2, 2]
    cnt = 0
    time = []
    #file_location = 'Results/test/'
    file_location = 'new_results/'
    plotter = Plotter(None, None, file_location)
    data_reader = DataReader()
    for dataset in datasets:
        if cnt < 0:
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
