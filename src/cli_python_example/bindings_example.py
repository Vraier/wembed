import argparse
import wembed

def main():

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Process some graph embeddings.')
    parser.add_argument('-i', '--input', required=True, help='Input graph file path')
    parser.add_argument('-o', '--output', required=False, help='Output embedding file path')

    parser.add_argument('--seed', type=int, default=1, help='Seed used during embedding. -1 uses time as seed')
    parser.add_argument('--layered', action='store_true', help='Uses a layered embedding')

    parser.add_argument('--dim', type=int, default=4, help='Embedding dimensions')
    parser.add_argument('--dim-hint', type=int, default=4, help='Embedding dimensions hint')
    parser.add_argument('--iterations', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--speed', type=float, default=10, help='Speed during gradient descent')
    parser.add_argument('--cooling', type=float, default=0.99, help='Cooling during gradient descent')

    args = parser.parse_args()

    if (args.seed != -1):
        wembed.setSeed(args.seed)

    graph_file_path = args.input
    embedding_file_path = args.output

    # read in graph
    graph = wembed.readEdgeList(graph_file_path)
    if not wembed.isConnected(graph):
        print("Graph is not connected")
        return

    # set embedder options and build embedder
    embedder_options = wembed.EmbedderOptions()
    embedder_options.embeddingDimension = args.dim
    embedder_options.dimensionHint = args.dim_hint
    embedder_options.maxIterations = args.iterations
    embedder_options.speed = args.speed
    embedder_options.cooling = args.cooling

    # construct simple or layered embedder
    embedder = None
    if not args.layered:
        embedder = wembed.Embedder(graph, embedder_options)
    else:
        coarsen_options = wembed.PartitionerOptions()
        edge_weights = [1] * (graph.getNumEdges() * 2)
        coarsener = wembed.LabelPropagation(coarsen_options, graph, edge_weights)
        embedder = wembed.LayeredEmbedder(graph, coarsener, embedder_options)

    # calculate embedding
    embedder.calculateEmbedding()

    # write embedding to file
    if embedding_file_path is not None:
        wembed.writeCoordinates(embedding_file_path, embedder.getCoordinates(), embedder.getWeights())
        

if __name__ == "__main__":
    main()
