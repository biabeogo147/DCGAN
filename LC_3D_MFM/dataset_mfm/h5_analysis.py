import h5py
import torch
from pytorch3d.io import save_obj


def review_h5_file(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        def print_hierarchy(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Data type: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        h5_file.visititems(print_hierarchy)

        if 'my_dataset' in h5_file:
            data = h5_file['my_dataset'][:]
            print("Data in 'my_dataset':", data)


def print_first_line(file_path, name_group_or_dataset):
    with h5py.File(file_path, 'r') as h5_file:
        if name_group_or_dataset in h5_file:
            obj = h5_file[name_group_or_dataset]
            if isinstance(obj, h5py.Dataset):
                print(f"First line of dataset '{name_group_or_dataset}': {obj[0]}")
            elif isinstance(obj, h5py.Group):
                keys = list(obj.keys())
                if keys:
                    print(f"First item in group '{name_group_or_dataset}': {keys[0]}")
                else:
                    print(f"Group '{name_group_or_dataset}' is empty.")
            else:
                print("Invalid input, please provide a group or dataset name")
        else:
            print(f"'{name_group_or_dataset}' not found in the file.")


def save_mesh(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        shape_points = h5_file['color/representer/points'][:]
        shape_cells = h5_file['color/representer/cells'][:]
        color_mean = h5_file['color/model/mean'][:]

    vertices = torch.tensor(shape_points.T, dtype=torch.float32)
    faces = torch.tensor(shape_cells.T, dtype=torch.int64)
    colors = torch.tensor(color_mean.reshape(-1, 3) / 255.0, dtype=torch.float32)

    save_obj("D:/DS-AI/data/bfm2019_face_color.obj", vertices, faces)


if __name__ == '__main__':
    file_path = 'D:/DS-AI/data/model2019_bfm.h5'
    review_h5_file(file_path)
    print_first_line(file_path, 'color/representer/colorspace')
    save_mesh(file_path)


