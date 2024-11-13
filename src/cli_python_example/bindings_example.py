import argparse
import wembed

def main():

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Process some graph embeddings.')
    parser.add_argument('-i', '--input', required=True, help='Input graph file path')
    parser.add_argument('-o', '--output', required=False, help='Output embedding file path')

    parser.add_argument('--dim', type=int, default=4, help='Embedding dimensions')
    parser.add_argument('--dim-hint', type=int, default=4, help='Embedding dimensions hint')
    
    args = parser.parse_args()
    
    graph_file_path = args.input
    embedding_file_path = args.output
    
    # read in graph
    graph = wembed.readEdgeList(graph_file_path)
    if not wembed.isConnected(graph):
        print("Graph is not connected")
        return
    
    print(graph)

    # set embedder options and build embedder
    embedder_options = wembed.EmbedderOptions()
    embedder_options.embeddingDimension = args.dim
    embedder_options.dimensionHint = args.dim_hint
    embedder_options.maxIterations = 1000
    embedder = wembed.Embedder(graph, embedder_options)

    # calculate embedding
    embedder.calculateEmbedding()

    # write embedding to file
    if embedding_file_path is not None:
        wembed.writeCoordinates(embedding_file_path, embedder.getCoordinates(), embedder.getWeights())

if __name__ == "__main__":
    main()