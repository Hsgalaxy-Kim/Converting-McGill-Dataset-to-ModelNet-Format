import numpy as np
import trimesh
import os
import re
from tqdm import tqdm
import math

def pc_normalize(pc):
    """
    Normalize the point cloud to have zero mean and be scaled to unit sphere.
    Args:
        pc (np.array): Point cloud data [N, 3].
    Returns:
        np.array: Normalized point cloud.
    """
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc /= max_dist
    return pc

def list_folders(path):
    """
    Returns a list of folders in a specific path.

    Args:
        path (str): Path
    
    Returns:
        list: Folder(class) name lists
    """
    try:
        # Filtering only directories on a given path
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return folders
    except FileNotFoundError:
        print(f"Path not found: {path}")
        return []
    except PermissionError:
        print(f"You do not have permission for the path: {path}")
        return []

base_path = "modelnet_type" # McGill file save path

os.makedirs(base_path, exist_ok=True)

path = "."  # McGill Folder
folders = list_folders(path)
folders.remove(base_path)
print(f"Folder lists: {folders}")

def natural_sort_key(key):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', key)]

files_in_folders = {}

# Creating a File List
for folder in folders:
    if os.path.exists(folder):  # Verify that the folder actually exists
        files = os.listdir(folder)  # Get a list of files in that folder
        files.sort(key=natural_sort_key)  # Sort the file list in ascending order
        files_in_folders[folder] = files
    else:
        print(f"Folder '{folder}'does not exist.")

# Create subfolders
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    os.makedirs(folder_path, exist_ok=True)  # create folder

# Convert McGill to ModelNet format
for folder in tqdm(folders):
    for i, ply in enumerate(files_in_folders[folder]):
        ply_path = os.path.join(folder, ply)
        mesh = trimesh.load(ply_path, force="mesh")
        mesh.vertices = pc_normalize(mesh.vertices)
        vertices = mesh.vertices
        normals = mesh.vertex_normals
        array = np.concatenate((vertices, normals), axis=1)
        file_name = f"{folder}_{i:04d}.txt"
        target_path = os.path.join(base_path, folder, file_name)
        np.savetxt(target_path, array, delimiter=",", fmt="%.6f")

files_in_target_folders = {}

# Creating a File List
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if os.path.exists(folder_path):  # Verify that the folder actually exists
        files = os.listdir(folder_path)  # Get a list of files in that folder
        files.sort(key=natural_sort_key)  # Sort the file list in ascending order
        files_in_target_folders[folder] = files
    else:
        print(f"Folder '{folder}'does not exist.")

# Create train.txt and test.txt files
with open(f"{base_path}/train.txt", "w") as train_file, open(f"{base_path}/test.txt", "w") as test_file, open(f"{base_path}/mcgill_shape_names.txt", "w") as name_file:
    for folder in tqdm(folders):
        files = files_in_target_folders[folder]
        split_index = 2 * len(files) / 3
        split_index = int(split_index)
        train_files = files[:split_index]
        test_files = files[split_index:]

        for file in train_files:
            train_file.write(f"{file}\n")

        for file in test_files:
            test_file.write(f"{file.split('.')[0]}\n")
        
        name_file.write(f"{folder}\n")

        print(f"Files in the folder '{folder}' have been written to train.txt and test.txt.")