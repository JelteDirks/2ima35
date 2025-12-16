import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass


@dataclass
class Vertex:
    """Represents a vertex in the graph with its features and metadata."""
    id: int
    features: np.ndarray
    label: float

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Vertex) and self.id == other.id

    def __repr__(self):
        return f"Vertex(id={self.id}, label={self.label})"


@dataclass
class Edge:
    """Represents a weighted edge between two vertices."""
    vertex1: Vertex
    vertex2: Vertex
    weight: float

    # hash (u,v) and (v,u) the same
    def __hash__(self):
        return hash((min(self.vertex1.id,
                         self.vertex2.id), max(self.vertex1.id,
                                               self.vertex2.id)))

    def __repr__(self):
        return f"Edge({self.vertex1.id} <-> {self.vertex2.id}, weight={self.weight:.4f})"

    def __lt__(self, other):
        """Enable sorting edges by weight for MST algorithms."""
        return self.weight < other.weight


class GraphBuilder:
    """
    Configurable graph builder for creating vertex and edge sets from datasets.
    Designed for MPC-MST algorithms with flexible feature selection.
    """

    def __init__(self,
                 filepath: str,
                 feature_columns: Optional[List[str]] = None,
                 use_all_features: bool = True,
                 max_samples: Optional[int] = None):
        """
        Initialize the graph builder.
        Args:
            filepath: Path to the CSV file
            feature_columns: Specific feature columns to use (if None and use_all_features=False, uses f1-f8)
            use_all_features: If True, uses all feature columns except 'signal'
            max_samples: Maximum number of samples to load (None = all samples)
        """
        self.filepath = filepath
        self.feature_columns = feature_columns
        self.use_all_features = use_all_features
        self.max_samples = max_samples

        self.df = None
        self.vertices = None
        self.edges = None
        self.scaler = None
        self.selected_features = None

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset from disk with optional sample limit."""
        print(f"Loading dataset from: {self.filepath}")

        if self.max_samples:
            self.df = pd.read_csv(self.filepath, nrows=self.max_samples)
            print(
                f"  Loaded {len(self.df)} samples (limited to {self.max_samples})"
            )
        else:
            self.df = pd.read_csv(self.filepath)
            print(f"  Loaded {len(self.df)} samples")

        print(f"  Columns: {list(self.df.columns)}")
        return self.df

    def select_features(self) -> List[str]:
        """
        Select which features to use based on configuration.
        Returns:
            List of feature column names to use
        """
        all_columns = self.df.columns.tolist()

        if self.use_all_features:
            # Use all columns except 'signal'
            self.selected_features = [
                col for col in all_columns if col != 'signal'
            ]
        elif self.feature_columns:
            # Use user-specified features
            self.selected_features = self.feature_columns
            # Validate that columns exist
            missing = set(self.selected_features) - set(all_columns)
            if missing:
                raise ValueError(f"Features not found in dataset: {missing}")
        else:
            # Default: use only f1-f8 features
            self.selected_features = [
                col for col in all_columns if col.startswith('f')
            ]

        print(
            f"\nSelected {len(self.selected_features)} features: {self.selected_features}"
        )
        return self.selected_features

    def create_vertices(self) -> Set[Vertex]:
        """
        Create vertex set from the dataset.
        Returns:
            Set of Vertex objects
        """
        if self.df is None:
            self.load_dataset()

        if self.selected_features is None:
            self.select_features()

        # Extract features and labels
        labels = self.df['signal'].values
        raw_features = self.df[self.selected_features].values

        # Create vertex objects
        self.vertices = set()
        for i in range(len(self.df)):
            vertex = Vertex(id=i, features=raw_features[i], label=labels[i])
            self.vertices.add(vertex)

        print(f"\nCreated {len(self.vertices)} vertices")
        return self.vertices

    def compute_distance(self,
                         vertex1: Vertex,
                         vertex2: Vertex,
                         metric: str = 'euclidean') -> float:
        """
        Helper function to compute distance between two vertices.
        Args:
            vertex1: First vertex
            vertex2: Second vertex
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        Returns:
            Distance between the two vertices
        """
        if metric == 'euclidean':
            return np.sqrt(np.sum((vertex1.features - vertex2.features)**2))
        elif metric == 'manhattan':
            return np.sum(np.abs(vertex1.features - vertex2.features))
        elif metric == 'cosine':
            dot_product = np.dot(vertex1.features, vertex2.features)
            norm1 = np.linalg.norm(vertex1.features)
            norm2 = np.linalg.norm(vertex2.features)
            return 1 - (dot_product / (norm1 * norm2))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def create_edges(self,
                     complete: bool = True,
                     max_edges: Optional[int] = None,
                     distance_threshold: Optional[float] = None,
                     metric: str = 'euclidean') -> Set[Edge]:
        """
        Create edge set from vertices.
        Args:
            complete: If True, creates complete graph (all possible edges)
            max_edges: Maximum number of edges to create (None = unlimited)
            distance_threshold: Only create edges with distance <= threshold
            metric: Distance metric to use
        Returns:
            Set of Edge objects
        """
        if self.vertices is None:
            self.create_vertices()

        print("\nCreating edges...")
        print(f"  Complete graph: {complete}")
        print(f"  Distance metric: {metric}")
        if distance_threshold:
            print(f"  Distance threshold: {distance_threshold}")

        self.edges = set()
        vertex_list = list(self.vertices)
        n = len(vertex_list)

        # Generate all possible edges
        edge_count = 0
        total_possible = n * (n - 1) // 2

        for i, v1 in enumerate(vertex_list):
            for v2 in vertex_list[i + 1:]:
                if max_edges and edge_count >= max_edges:
                    break

                # Compute distance
                distance = self.compute_distance(v1, v2, metric=metric)

                # Apply threshold filter if specified
                if distance_threshold is None or distance <= distance_threshold:
                    edge = Edge(vertex1=v1, vertex2=v2, weight=distance)
                    self.edges.add(edge)
                    edge_count += 1

            if max_edges and edge_count >= max_edges:
                break

            # Progress indicator for large graphs
            if (i + 1) % 100 == 0:
                print(
                    f"  Processed {i+1}/{n} vertices, created {edge_count} edges"
                )

        print(
            f"\nCreated {len(self.edges)} edges out of {total_possible} possible"
        )
        return self.edges

    def get_graph_summary(self) -> Dict:
        """
        Get summary statistics about the constructed graph.
        Returns:
            Dictionary with graph statistics
        """
        if self.vertices is None or self.edges is None:
            raise ValueError(
                "Graph not yet constructed. Call create_vertices() and create_edges() first."
            )

        edge_weights = [e.weight for e in self.edges]

        summary = {
            'num_vertices':
            len(self.vertices),
            'num_edges':
            len(self.edges),
            'num_features':
            len(self.selected_features),
            'feature_names':
            self.selected_features,
            'min_edge_weight':
            min(edge_weights) if edge_weights else None,
            'max_edge_weight':
            max(edge_weights) if edge_weights else None,
            'mean_edge_weight':
            np.mean(edge_weights) if edge_weights else None,
            'median_edge_weight':
            np.median(edge_weights) if edge_weights else None,
        }

        return summary

    def build_graph(
            self,
            complete: bool = True,
            max_edges: Optional[int] = None,
            distance_threshold: Optional[float] = None,
            metric: str = 'euclidean') -> Tuple[Set[Vertex], Set[Edge]]:
        """
        Convenience method to build the complete graph in one call.
        Returns:
            Tuple of (vertices, edges)
        """
        self.load_dataset()
        self.select_features()
        self.create_vertices()
        self.create_edges(complete=complete,
                          max_edges=max_edges,
                          distance_threshold=distance_threshold,
                          metric=metric)

        print("\n" + "=" * 70)
        print("GRAPH CONSTRUCTION COMPLETE")
        print("=" * 70)

        summary = self.get_graph_summary()
        for key, value in summary.items():
            if isinstance(value, (list, tuple)) and len(value) > 5:
                print(f"{key}: {value[:5]}... ({len(value)} total)")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        return self.vertices, self.edges


# Example usage demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("GRAPH BUILDER FOR MPC-MST - EXAMPLE USAGE")
    print("=" * 70)

    print("\n\n### Example 2: Graph with selected features ###")
    builder = GraphBuilder(filepath='~/Downloads/SUSY.csv',
                           use_all_features=False,
                           feature_columns=['f1', 'f2'],
                           max_samples=500)
    vertices, edges = builder.build_graph(complete=True)

    # Demonstrate distance computation
    print("\n\n### Distance Computation Examples ###")
    v_list = list(vertices)
    if len(v_list) >= 2:
        v1, v2 = v_list[0], v_list[1]
        print(f"\nDistance between {v1} and {v2}:")
        print(
            f"  Euclidean: {builder.compute_distance(v1, v2, 'euclidean'):.4f}"
        )
        print(
            f"  Manhattan: {builder.compute_distance(v1, v2, 'manhattan'):.4f}"
        )
        print(f"  Cosine: {builder.compute_distance(v1, v2, 'cosine'):.4f}")
