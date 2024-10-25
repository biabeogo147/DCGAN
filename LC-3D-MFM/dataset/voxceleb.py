import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class VOXCeleb3dDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.root_dir = root_dir
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".npy"):
                        file_path = os.path.join(folder_path, file_name)
                        self.samples.append(file_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        folder_path = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        npy_data = np.load(file_path)

        xyz_path = os.path.join(folder_path, f"{base_name}.xyz")
        with open(xyz_path, 'r') as xyz_file:
            xyz_data = xyz_file.readlines()

        image_path = os.path.join(folder_path, f"{base_name}.jpg")
        image = Image.open(image_path).convert('RGB')

        return {
            'npy_data': npy_data,
            'xyz_data': xyz_data,
            'image': image
        }
