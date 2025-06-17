import os
import torch
import json

# used for data saved by save_tensor_shapes.py

def inspect_samples(samples_path):
    entries = os.listdir(samples_path)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(samples_path, entry))]
    att_shapes = set()
    index_shapes = set()
    print(f"\nInspecting {len(folders)} samples in: {samples_path}")
    print("=" * 80)
    for i, folder in enumerate(folders):
        folder_path = os.path.join(samples_path, folder)
        index_tensor = torch.load(os.path.join(folder_path, 'index_tensor.pt'))
        with open(os.path.join(folder_path, 'shapes.json'), "r") as f:
            shape_data = json.load(f)

        print(f"\nSample {i+1}/{len(folders)} - {folder}:")
        print(f"  Input shape: {shape_data['att_shape']}")
        print(f"  Index shape:     {shape_data['index_shape']}")

        assert list(index_tensor.shape) == shape_data['index_shape'], "loaded tensor shape should be equal to the one saved in json file"
        att_shapes.add(tuple(shape_data['att_shape']))
        index_shapes.add(tuple(shape_data['index_shape']))

    print("\n" + "=" * 80)
    print("\nUnique Shapes Summary:")
    print(f"\nInput shapes ({len(att_shapes)} unique):")
    for shape in sorted(att_shapes):
        print(f"  {shape}")
    
    print(f"\nIndex shapes ({len(index_shapes)} unique):")
    for shape in sorted(index_shapes):
        print(f"  {shape}")

    return {
        'input_shapes': att_shapes,
        'index_shapes': index_shapes,
    }

if __name__ == "__main__":
    inspect_samples('sample_data')