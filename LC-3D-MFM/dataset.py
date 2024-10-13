from utils import *


class VOXCeleb3dDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        super().__init__()
        self.ext_dict = {
            '.npy': 'npy',
            '.xyz': 'xyz',
            '.jpg_b': 'b_image',
            '.jpg_s': 's_image'
        }
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.data_list = self._load_dataset()

    def _load_dataset(self):
        dataset = []

        for folder in os.listdir(self.dataset_dir):
            sample = dict()
            for file_name in os.listdir(folder):
                base_name, ext = os.path.splitext(file_name)
                sample_id = base_name.split('_')[0]
                sample['sample_id'] = sample_id

                key = self.ext_dict.get(ext if ext != '.jpg' else f'{ext}_{base_name[-1]}')
                sample[key] = os.path.join(self.dataset_dir, folder, file_name)
        return dataset

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        npy_data = np.load(sample[self.ext_dict['.npy']])
        xyz_data = np.loadtxt(sample[self.ext_dict['.xyz']])
        b_image = Image.open(sample[self.ext_dict['.jpg_b']]).convert('RGB')
        s_image = Image.open(sample[self.ext_dict['.jpg_s']]).convert('RGB')

        if self.transform:
            b_image = self.transform(b_image)
            s_image = self.transform(s_image)

        return {
            'npy_data': torch.tensor(npy_data, dtype=torch.float32),
            'xyz_data': torch.tensor(xyz_data, dtype=torch.float32),
            'b_image': b_image,
            's_image': s_image
        }
