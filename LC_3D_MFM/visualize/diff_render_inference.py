import torch
import matplotlib.pyplot as plt
from LC_3D_MFM.dataset_mfm.obj_analysis import load_obj
from LC_3D_MFM.model import  image_formation_matplotlib
from LC_3D_MFM.dataset_mfm.h5_analysis import get_face_properties_from_h5


def save_image(image):
    plt.imsave("../../testing_landmark_detect/image.png", image)


def plot_rgbd_image(image, title=""):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def mesh_no_texture(vertices, faces, title=""):
    poses = torch.tensor([[-0.2, 0.0, 0.0, 0.0, 0.0, 1.75]], dtype=torch.float32)
    # poses = torch.tensor([[0.0, -1.57, 0.0, 0.0, 0.0, 2.0]], dtype=torch.float32)

    light_positions = torch.tensor([
        [5.0, 5.0, 5.0],  # Nguồn sáng 1
        [-5.0, 5.0, 5.0],  # Nguồn sáng 2
        [0.0, -5.0, 5.0],  # Nguồn sáng 3
    ], dtype=torch.float32)

    project_function = image_formation_matplotlib.DifferentialRender(image_size=240, device="cpu")
    image = project_function(vertices, faces, torch.zeros_like(vertices), poses, light_positions)
    image = image.squeeze(1).squeeze(0).cpu().numpy()
    plot_rgbd_image(image, title)


def plot_colored_points(vertices, colors, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=colors, s=2)
    ax.view_init(90, -90)
    plt.title(title)
    plt.axis('off')
    plt.show()


def mesh_with_texture(vertices, faces, colors, title=""):
    pose = torch.tensor([[-0.2, 0.0, 0.0, 0.0, 0.0, 2.0]], dtype=torch.float32)
    # pose = torch.tensor([[0.0, -1.57, 0.0, 0.0, 0.0, 2.0]], dtype=torch.float32)
    project_function = image_formation_matplotlib.DifferentialRender(image_size=240, device="cpu")
    image = project_function(vertices, faces, colors, pose)
    image = image.squeeze(1).squeeze(0).cpu().numpy()
    plot_rgbd_image(image, title)


if __name__ == '__main__':
    # file_path_obj = "../../data/male.obj"
    file_path_obj = "D:/DS-AI/data/voxceleb3d/all.obj"
    vertices, faces, colors = load_obj(file_path_obj)
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int32)
    colors = torch.tensor(colors, dtype=torch.float32)
    # plot_pointcloud(vertices, faces, title="Original Mesh")
    mesh_no_texture(vertices, faces, title="Mesh without Texture")

    # file_path_h5 = "D:/DS-AI/data/model2019_bfm.h5"
    # vertices, faces, colors = get_face_properties_from_h5(file_path_h5)
    # print(vertices.shape, faces.shape, colors.shape)
    # plot_colored_points(vertices, colors, title="Colored Points")
    # mesh_with_texture(vertices, faces, colors, title="Mesh with Texture")
