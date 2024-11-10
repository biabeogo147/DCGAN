import matplotlib.pyplot as plt
import torch

from LC_3D_MFM.model import image_formation
from LC_3D_MFM.dataset_mfm.mesh_analysis import count_vertices_and_faces


def get_face_vertices(vertices, faces):
    face_vertices = [[vertices[i - 1] for i in face] for face in faces]
    return face_vertices


if __name__ == '__main__':
    file_path = "../../data/male.obj"

    all_vertices, all_faces, _, _ = count_vertices_and_faces(file_path)
    pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 25.0], dtype=torch.float32)

    project_function = image_formation.ProjectFunction()
    image = project_function(all_vertices, all_faces, pose)

    print(image.shape)

    plt.imshow(image.cpu().numpy())
    plt.axis("off")
    plt.show()


