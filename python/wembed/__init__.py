from ._core import (
    __doc__,
    __version__,
    Graph,
    EmbedderOptions,
    EmbedderInterface,
    Embedder,
    LayeredEmbedder,
    LabelPropagation,
    PartitionerOptions,
    readEdgeList,
    writeCoordinates,
    isConnected,
    setSeed,
)

def convert_from_networkx_graph(graph):
    """Converts a NetworkX graph to a WEmbed-compatible Graph object."""
    edges = list(graph.edges)
    edge_ids = set()
    
    for edge in edges:
        if not isinstance(edge[0], int) or not isinstance(edge[1], int):
            raise ValueError("Edge ids must be integers")
        edge_ids.add(edge[0])
        edge_ids.add(edge[1])
    
    if edge_ids != set(range(len(edge_ids))):
        raise ValueError("Edge ids must be consecutive and start from 0")
    
    return _core.Graph(edges)