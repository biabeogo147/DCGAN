import os
from PIL import Image
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    self.samples.append(os.path.join(folder_path, file_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        image = Image.open(file_path).convert('RGB')
        image = self.transform(image) if self.transform else image

        return image, 0
