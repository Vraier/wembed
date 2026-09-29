import argparse
import time

import wembed


def main():
    parser = argparse.ArgumentParser(description='Compute a weighted graph embedding.')
    parser.add_argument('-i', '--input', required=True, help='Input graph file path')
    parser.add_argument('-o', '--output', required=False, help='Output embedding file path')

    parser.add_argument('--seed', type=int, default=1, help='Seed used during embedding. -1 uses time as seed')
    parser.add_argument('--layered', action='store_true', help='Use a layered embedding')

    parser.add_argument('--dim', type=int, default=4, help='Embedding dimensions')
    parser.add_argument('--dim-hint', type=float, default=4, help='Embedding dimensions hint')
    parser.add_argument('--iterations', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--speed', type=float, default=10, help='Learning rate for gradient descent')
    parser.add_argument('--cooling', type=float, default=0.99, help='Cooling factor for gradient descent')
    parser.add_argument('--centre', type=float, default=0.0,
                        help='Strength of centre-pull force (useful for unconnected graphs)')

    args = parser.parse_args()

    if args.seed != -1:
        wembed.setSeed(args.seed)

    graph = wembed.graphFromEdgeListFile(args.input)

    options = wembed.Options()
    options.embeddingDimension = args.dim
    options.dimensionHint = args.dim_hint
    options.maxIterations = args.iterations
    options.learningRate = args.speed
    options.lrCoolingFactor = args.cooling
    options.layeredEmbedding = args.layered
    options.centreScale = args.centre

    embedder = wembed.createEmbedder(graph, options)
    embed_with_progress(embedder)

    print(wembed.timingsToString(embedder.getTimings()))

    if args.output is not None:
        embedder.writeCoordinates(args.output)


def embed_with_progress(embedder, report_every=30.0):
    """Step through the embedding and print its progress every report_every seconds.

    embedder.calculateEmbedding() does the same silently.
    """
    last_report = time.monotonic()
    while not embedder.isFinished():
        embedder.calculateStep()
        if time.monotonic() - last_report >= report_every:
            p = embedder.getProgress()
            line = f"layer {p.layer}, iteration {p.iteration}"
            if p.expectedIterations >= 0:
                line += f" of ~{p.expectedIterations}"
            if p.etaSeconds >= 0:
                line += f", ETA {p.etaSeconds:.0f}s"
            print(line)
            last_report = time.monotonic()


def graph_from_networkx(nx_graph):
    """Convert a networkx graph to a wembed.Graph.

    Requires vertex IDs to be consecutive integers starting at 0.
    """
    edges = list(nx_graph.edges)
    vertex_ids = set()

    for u, v in edges:
        if not isinstance(u, int) or not isinstance(v, int):
            raise ValueError("Edge endpoints must be integers")
        vertex_ids.add(u)
        vertex_ids.add(v)

    if vertex_ids != set(range(len(vertex_ids))):
        raise ValueError("Vertex ids must be consecutive and start from 0")

    return wembed.graphFromEdges([wembed.Edge(u, v) for u, v in edges])


if __name__ == "__main__":
    main()
