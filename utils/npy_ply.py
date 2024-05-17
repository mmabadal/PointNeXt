import os
import numpy as np
from plyfile import PlyData, PlyElement

# Define basic colors
basic_colors = {
    0: (255, 0, 0),      # Red
    1: (0, 255, 0),      # Green
    2: (0, 0, 255),      # Blue
    3: (255, 255, 0),    # Yellow
    4: (255, 165, 0),    # Orange
    5: (128, 0, 128),    # Purple
    6: (0, 128, 128),    # Teal
    7: (255, 192, 203),  # Pink
    8: (128, 128, 128),  # Gray
    9: (0, 0, 0)         # Black
}

def generate_color_label_map(labels):
    color_label_map = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for label in np.unique(labels):
        mask = labels == label
        color_label_map[mask] = basic_colors.get(label % len(basic_colors))
    return color_label_map

def main(input_file):
    # Load point cloud data from numpy file
    data = np.load(input_file)
    points = data[:, :3]
    colors = data[:, 3:6]
    labels = data[:, 6]

    # Generate color for each point based on label
    color_label_map = generate_color_label_map(labels)

    # Get base path of input file
    base_path = os.path.dirname(input_file)

    # Write position and color PLY file
    vertex = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex['red'] = colors[:, 0]
    vertex['green'] = colors[:, 1]
    vertex['blue'] = colors[:, 2]
    el = PlyElement.describe(vertex, 'vertex')
    ply_filename = os.path.join(base_path, 'position_color.ply')
    PlyData([el]).write(ply_filename)

    # Write label-colored PLY file
    vertex_label = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_label['x'] = points[:, 0]
    vertex_label['y'] = points[:, 1]
    vertex_label['z'] = points[:, 2]
    vertex_label['red'] = color_label_map[:, 0]
    vertex_label['green'] = color_label_map[:, 1]
    vertex_label['blue'] = color_label_map[:, 2]
    el_label = PlyElement.describe(vertex_label, 'vertex')
    ply_label_filename = os.path.join(base_path, 'label_color.ply')
    PlyData([el_label]).write(ply_label_filename)

if __name__ == "__main__":
    input_file = "/home/miguel/Desktop/PIPES/dataset/3/npy/1710331046843080.npy"
    main(input_file)
