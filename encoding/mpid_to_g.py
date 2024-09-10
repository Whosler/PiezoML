import json
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from mendeleev import element
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
import copy
import pickle


# Load element data from file
with open('element_data.json', 'r') as f:
    element_data = json.load(f)


# Function to get element vector from local data
def get_element_vector(element_abbr):
    elem = element_data.get(element_abbr, {})
    number = elem.get('atomic_number', 0)
    group = float(elem.get('group_id', 0.0))
    mass = elem.get('atomic_weight', 0.0)
    polarizability = elem.get('dipole_polarizability', 0.0)
    electronegativity = elem.get('en_allen', 0.0)
    electron_affinity = elem.get('electron_affinity', 0.0)
    c6_coefficient = elem.get('c6_gb', 0.0)
    return np.array([number, group, mass, polarizability, electronegativity, electron_affinity, c6_coefficient])

# Function to read and parse JSON data from file
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Encode crystal systems
def encode_crystal_systems(data):
    crys_sys_list = list(set(compound["cry_sys"] for compound in data.values()))
    encoder = OneHotEncoder()
    encoder.fit(np.array(crys_sys_list).reshape(-1, 1))
    return encoder

def get_median_dij(json_dict):
    top_percentile = 0.1
    # Extract piezoelectric_constant values
    piezoelectric_constants = []
    for key, value in json_dict.items():
        if key.startswith("mp-") and "piezoelectric_constant" in value:
            piezoelectric_constants.append(value["piezoelectric_constant"])
    
    # Calculate and return the threshold for the top 20%
    if piezoelectric_constants:
        # Sort the constants in ascending order
        piezoelectric_constants.sort()
        # Calculate the index for the 80th percentile
        threshold_index = int((1-top_percentile) * len(piezoelectric_constants))
        threshold = piezoelectric_constants[threshold_index]
        print(threshold)
        return threshold
    else:
        return None  # or raise an error if appropriate


# Function to create NetworkX graphs from JSON data
def parse_json_to_nx(data, encoder, thresh):
    graphs = []
    unique_elements = set()
    n_high = 0
    n_low = 0
    
    for compound_id, compound_data in data.items():
        G = nx.Graph()
        elements = compound_data["elements"]
        coords = np.array(compound_data["cartesian_coords"])
        
        # Track unique elements
        unique_elements.update(elements)
        
        # Encode crystal system
        crys_sys_encoded = encoder.transform([[compound_data["cry_sys"]]])
        crys_sys_encoded = crys_sys_encoded.toarray().flatten()  # Convert to dense array and flatten
        
        # Add nodes with features (element vectors + crystal system)
        for i, element in enumerate(elements):
            feature_vector = get_element_vector(element)
            node_features = np.concatenate([feature_vector, crys_sys_encoded])
            G.add_node(i, element=element, features=node_features)
        
        # Add edges with weights (distances)
        num_atoms = len(elements)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distance = np.linalg.norm(coords[i] - coords[j])
                G.add_edge(i, j, weight=distance)
        
        # Convert piezoelectric constant to binary class
        piezoelectric_constant = compound_data["piezoelectric_constant"]
        class_label = 1 if piezoelectric_constant > thresh else 0
        n_high += class_label
        n_low += 1 if class_label == 0 else 0
        G.graph["piezoelectric_class"] = class_label
        G.graph["compound_id"] = compound_id
        graphs.append(G)
    
    print(f"n_high: {n_high}")
    print(f"n_low: {n_low}")
    print("Unique elements in dataset:", unique_elements)
    
    return graphs, unique_elements

# Function to convert NetworkX graphs to PyTorch Geometric Data objects
def nx_to_data(nx_graph):
    data = from_networkx(nx_graph)
    features = np.array([nx_graph.nodes[n]['features'] for n in nx_graph.nodes])
    data.x = torch.tensor(features, dtype=torch.float)
    
    # Ensure target tensor is 1D for classification
    label = torch.tensor([nx_graph.graph["piezoelectric_class"]], dtype=torch.long).view(-1)
    data.y = label

    data.compound_id = nx_graph.graph["compound_id"]
    
    return data

def find_zero_fields(element_abbr):
    field_names = ["atomic_number", "group_id", "atomic_weight", "dipole_polarizability", "en_allen", "electron_affinity", "c6_gb"]
    vector = get_element_vector(element_abbr)
    zero_fields = [field_names[i] for i, value in enumerate(vector) if value == 0.0]
    return zero_fields

# Read JSON data from file
file_path = '../../../Raw_data_0814/JAV_piezo_dij.json'
json_data = read_json_file(file_path)

median = get_median_dij(json_data)

# Encode crystal systems
encoder = encode_crystal_systems(json_data)
with open('onehot_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Parse JSON data to create NetworkX graphs
graphs, unique_elements = parse_json_to_nx(json_data, encoder, median)

for element in unique_elements:
    zero_fields = find_zero_fields(element)
    print(f"Element: {element}, Zero Fields: {zero_fields}")

# Convert NetworkX graphs to PyTorch Geometric Data objects
data_list = [nx_to_data(graph) for graph in graphs]

with open('../../Training/Piezo_training/dij_data_list_0.1.pkl', 'wb') as f:
    pickle.dump(data_list, f)
