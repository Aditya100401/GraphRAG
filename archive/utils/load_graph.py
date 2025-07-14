import pickle

def load_graph(file_path):
    """
    Load a NetworkX graph from a pickle file.
    
    Parameters:
    -----------
    file_path : str
        Path to the pickle file containing the graph.
        
    Returns:
    --------
    networkx.Graph or derived class
        The loaded graph.
    """
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph
