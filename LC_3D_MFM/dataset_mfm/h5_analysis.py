import h5py
import torch
from pytorch3d.io import save_obj
from matplotlib import pyplot as plt
from LC_3D_MFM.model import image_formation


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


def plot_colored_points(vertices, colors, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=colors, s=2)
    ax.view_init(90, -90)
    plt.title(title)
    plt.axis('off')
    plt.show()


def mesh_with_texture(vertices, faces, colors, title=""):
    pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.5], dtype=torch.float32)
    project_function = image_formation.ProjectFunction(focal_length=200.0, image_size=240, device="cpu")
    image = project_function(vertices, faces, colors, pose)
    plt.imshow(image.cpu().numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_h5(file_path, is_save_obj=False):
    with h5py.File(file_path, 'r') as h5_file:
        shape_points = h5_file['color/representer/points'][:]
        shape_cells = h5_file['color/representer/cells'][:]
        color_mean = h5_file['color/model/mean'][:]

    vertices = torch.tensor(shape_points.T, dtype=torch.float32).to(device)
    faces = torch.tensor(shape_cells.T, dtype=torch.int64).to(device)
    colors = torch.tensor(color_mean.reshape(-1, 3), dtype=torch.float32).to(device)

    if is_save_obj:
        save_obj("D:/DS-AI/data/bfm2019_face_color.obj", vertices, faces)

    plot_colored_points(vertices, colors, title="BFM 2019 Face Color")
    mesh_with_texture(vertices, faces, colors, title="BFM 2019 Face Color")


if __name__ == '__main__':
    file_path = 'D:/DS-AI/data/model2019_bfm.h5'
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # review_h5_file(file_path)
    # print_first_line(file_path, 'color/representer/colorspace')
    plot_h5(file_path)


