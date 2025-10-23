import pickle
import numpy as np

def inspect_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded type: {type(data)}")

    # If it's a list or tuple, iterate through elements
    if isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            if hasattr(item, 'shape'):
                print(f"Element {i}: shape = {item.shape}")
            else:
                print(f"Element {i}: type = {type(item)}")
    
    # If it's a dict, print shapes/types of its values
    elif isinstance(data, dict):
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(f"Key '{k}': shape = {v.shape}")
            else:
                print(f"Key '{k}': type = {type(v)}")
                print(v)
    
    # If it's a single array or other object
    else:
        if hasattr(data, 'shape'):
            print(f"Data shape: {data.shape}")
        else:
            print(f"Data type: {type(data)}")

if __name__ == "__main__":
    path = "/home/yixuan/Downloads/all_data/data_tiger/demos/20251013T164928261434/data.pkl"  # Replace with your file path
    inspect_pickle(path)

