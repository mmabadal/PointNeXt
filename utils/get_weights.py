import os
import numpy as np

def count_labels_in_npy_files(folder_path):
    label_counts = {}

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            
            # Load numpy array
            data = np.load(file_path)
            labels = data[:, -1]  # Extract labels from the last column

            # Count labels in this numpy array
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                label_counts[label] = label_counts.get(label, 0) + count
    
    return label_counts

def total_label_counts(label_counts):
    total_counts = {}
    for label, count in label_counts.items():
        total_counts[label] = total_counts.get(label, 0) + count
    return total_counts

# Example usage
folder_path = "/home/miguel/PointNeXt/data/pipes/pool/split_1"
label_counts = count_labels_in_npy_files(folder_path)
total_counts = total_label_counts(label_counts)

print("Label Counts for Each .npy File:")
print(label_counts)
print("\nTotal Label Counts:")
print(total_counts)
