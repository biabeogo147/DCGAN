import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from LC_3D_MFM.model import image_formation
from pytorch3d.ops import sample_points_from_meshes
from LC_3D_MFM.dataset_mfm.h5_analysis import get_face_properties_from_h5


def get_face_vertices(vertices, faces):
    face_vertices = [[vertices[i - 1] for i in face] for face in faces]
    return face_vertices


def plot_pointcloud(mesh, title=""):
    points = sample_points_from_meshes(mesh, num_samples=len(all_vertices))
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(0, 180)
    plt.show()


def plot_colored_points(vertices, colors, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=colors, s=2)
    ax.view_init(90, -90)
    plt.title(title)
    plt.axis('off')
    plt.show()


def mesh_with_texture(vertices, faces, colors, title=""):
    pose = torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 1.55], dtype=torch.float32)
    project_function = image_formation.ProjectFunction(focal_length=200.0, image_size=240, device="cpu")
    image = project_function(vertices, faces, colors, pose)
    plt.imshow(image.cpu().numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    file_path_obj = "D:/DS-AI/data/voxceleb3d/all.obj"
    # file_path = "../../data/male.obj"

    all_vertices, all_faces, _ = load_obj(file_path_obj)
    all_faces = all_faces.verts_idx

    # test_mesh = Meshes(verts=[all_vertices], faces=[all_faces])
    # plot_pointcloud(test_mesh, title="Original Mesh")

    file_path_h5 = "D:/DS-AI/data/model2019_bfm.h5"
    vertices, faces, colors = get_face_properties_from_h5(file_path_h5)
    # plot_colored_points(vertices, colors, title="Colored Points")
    mesh_with_texture(vertices, faces, colors, title="Mesh with Texture")
