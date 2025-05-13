import os
import networkx as nx
import matplotlib.pyplot as plt

file_path = os.path.join(os.path.dirname(__file__), "file2.txt")

def read_graph_from_file(file_path):
    graph = nx.Graph()  # Convert to an undirected graph for feature extraction
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            from_node, to_node = map(int, line.split())
            graph.add_edge(from_node, to_node)
    
    return graph

def compute_graph_features(graph):
    features = {}
    
    for node in graph.nodes():
        degree = graph.degree(node)
        features[node] = {"degree": degree}
    
    common_neighbors = {}
    jaccard_coefficients = {}
    
    for u, v in nx.non_edges(graph):  # Compute for non-existing edges (potential future collaborations)
        common_neighbors[(u, v)] = len(list(nx.common_neighbors(graph, u, v)))
        
    for u, v, p in nx.jaccard_coefficient(graph):
        jaccard_coefficients[(u, v)] = p
    
    return features, common_neighbors, jaccard_coefficients

def export_features(features, common_neighbors, jaccard_coefficients, output_file):
    with open(output_file, 'w') as file:
        file.write("Node Features:\n")
        for node, data in features.items():
            file.write(f"Node {node}: Degree = {data['degree']}\n")
        
        file.write("\nCommon Neighbors:\n")
        for (u, v), count in common_neighbors.items():
            file.write(f"({u}, {v}): {count}\n")
        
        file.write("\nJaccard Coefficients:\n")
        for (u, v), coeff in jaccard_coefficients.items():
            file.write(f"({u}, {v}): {coeff:.4f}\n")

def export_adjacency_list(graph, output_file):
    with open(output_file, 'w') as file:
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            file.write(f"{node}: {', '.join(map(str, neighbors))}\n")

def visualize_graph(graph):
    # pos = nx.kamada_kawai_layout(graph)
    # pos = nx.spring_layout(graph, seed=42)   
    # pos = nx.spectral_layout(graph) 
    # pos = nx.circular_layout(graph)  
    pos = nx.planar_layout(graph)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=800, edgecolors="black")
    nx.draw_networkx_edges(graph, pos, alpha=0.5, edge_color="gray", arrows=True)
    nx.draw_networkx_labels(graph, pos, font_size=8)
    plt.title("Graph Visualization", fontsize=12)
    plt.show()
    



if __name__ == "__main__":
    features_output_file = "graph_features.txt"
    adjacency_output_file = "adjacency_list.txt"
    
    graph = read_graph_from_file(file_path)
    features, common_neighbors, jaccard_coefficients = compute_graph_features(graph)
    export_features(features, common_neighbors, jaccard_coefficients, features_output_file)
    export_adjacency_list(graph, adjacency_output_file)
    visualize_graph(graph)
    
    print("Feature extraction completed. Results saved in", features_output_file)
    print("Adjacency list saved in", adjacency_output_file)