import h5py
import numpy as np

def print_structure(name, obj):
    """Helper to print dataset or group structure."""
    indent_level = name.count('/')
    indent = '  ' * indent_level
    if isinstance(obj, h5py.Dataset):
        shape = obj.shape
        dtype = obj.dtype
        print(f"{indent}- Dataset: {name} | shape={shape}, dtype={dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}+ Group: {name}")

def preview_dataset(ds, max_items=5):
    """Preview the first few elements of a dataset."""
    try:
        data = ds[()]
        print(f"    Preview (first {max_items} items): {np.array2string(data.flatten()[:max_items], threshold=max_items)}")
    except Exception as e:
        print(f"    Could not preview dataset: {e}")

def inspect_hdf5(path, preview=False):
    """Read and interpret an HDF5 file, printing structure and dataset info."""
    with h5py.File(path, 'r') as f:
        print(f"Opened HDF5 file: {path}\n")
        f.visititems(print_structure)
        
        if preview:
            print("\n--- Dataset Previews ---")
            def show_preview(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}")
                    preview_dataset(obj)
                    import pdb; pdb.set_trace()
            f.visititems(show_preview)

if __name__ == "__main__":
    # Example usage
    path = "sim_demos_ground_red_1/sim_271_quaternion.hdf5"  # <-- replace with your file path
    inspect_hdf5(path, preview=True)

