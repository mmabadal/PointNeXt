import os
import numpy as np
from plyfile import PlyData
from natsort import natsorted

def process_ply_file(file_path, label):
    # Read PLY file
    plydata = PlyData.read(file_path)
    vertices = plydata['vertex']
    # Extract x, y, z, r, g, b attributes
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    r = vertices['red']
    g = vertices['green']
    b = vertices['blue']
    # Create numpy array with x, y, z, r, g, b, label
    data = np.column_stack((x, y, z, r, g, b, np.full_like(x, label)))
    return data

def generate_dataset(dataset_folder):
    for ply_file in natsorted(os.listdir(dataset_folder)):
        print("working on: " + str(ply_file))
        output_file = os.path.join(dataset_folder,ply_file[:-4] + '.npy')
        if ply_file.endswith('.ply'):
            ply_path = os.path.join(dataset_folder, ply_file)
            label = 0
            data = process_ply_file(ply_path, label)
        np.save(output_file, data)

# Example usage:
dataset_folder = '/home/miguel/Desktop/PIPES/dataset/3/ply'
dataset = generate_dataset(dataset_folder)



