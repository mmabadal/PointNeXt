import os
import numpy as np
from plyfile import PlyData
from natsort import natsorted

# Define class labels
class_labels = {
    'plastic': 0,
    'floor': 0,
    'block':0,
    'vessel':0,  
    'rock':0,
    'sand':0,
    'po':0, 
    'pipe': 1,
    'valve': 2
    # Add more class labels as needed
}

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
    for data_folder in natsorted(os.listdir(dataset_folder)):
        print("working on: " + str(data_folder))
        data_path = os.path.join(dataset_folder, data_folder)
        output_file = os.path.join(data_path,data_folder + '.npy')
        annotation_folder = os.path.join(data_path, 'annotations')
        data = []
        for annotation_file in natsorted(os.listdir(annotation_folder)):
            if annotation_file.endswith('.ply'):
                annotation_path = os.path.join(annotation_folder, annotation_file)
                label = 99
                for key, value in class_labels.items():
                    if key in annotation_file:
                        label = value
                        break
                
                if label==99:
                    print("no encuentro label para la clase:" + str(annotation_file))
                    
                data.append(process_ply_file(annotation_path, label))  
                stacked_data = np.vstack(data)          

        np.save(output_file, stacked_data)

# Example usage:
dataset_folder = '/home/miguel/Desktop/PIPES/dataset/2/sea3'
dataset = generate_dataset(dataset_folder)



