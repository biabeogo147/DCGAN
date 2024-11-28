import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class VOXCeleb3dDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            self.samples.append(folder_path)
            break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = []
        folder_path = self.samples[idx]
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.npy'):
                base_name, _ = os.path.splitext(file_name)

                npy_path = os.path.join(folder_path, f"{base_name}.npy")
                xyz_path = os.path.join(folder_path, f"{base_name}.xyz")
                image_path = os.path.join(folder_path, f"{base_name}_b.jpg")

                npy_data = np.load(npy_path, allow_pickle=True).item()
                xyz_data = np.loadtxt(xyz_path)

                image = Image.open(image_path).convert('RGB')
                image = self.transform(image) if self.transform else image

                data.append({
                    'npy_data': npy_data,
                    'xyz_data': xyz_data,
                    'image': image,
                })

        return data
