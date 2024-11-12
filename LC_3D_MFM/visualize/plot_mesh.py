import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from LC_3D_MFM.dataset_mfm.obj_analysis import count_vertices_and_faces


def get_face_vertices(vertices, faces):
    face_vertices = [[vertices[i] for i in face] for face in faces]
    return face_vertices


if __name__ == '__main__':
    file_path = "D:/DS-AI/data/voxceleb3d/all.obj"
    # file_path = "D:/DS-AI/data/bfm2019_face_color.obj"
    # file_path = "../../data/male.obj"

    all_vertices, all_faces, _, _ = count_vertices_and_faces(file_path)
    all_face_vertices = get_face_vertices(all_vertices, all_faces)

    all_vertices = np.array(all_vertices)
    all_faces = np.array(all_faces)

    max_range = np.array([all_vertices[:, 0].max() - all_vertices[:, 0].min(),
                          all_vertices[:, 1].max() - all_vertices[:, 1].min(),
                          all_vertices[:, 2].max() - all_vertices[:, 2].min()]).max() / 2.0

    mid_x = (all_vertices[:, 0].max() + all_vertices[:, 0].min()) * 0.5
    mid_y = (all_vertices[:, 1].max() + all_vertices[:, 1].min()) * 0.5
    mid_z = (all_vertices[:, 2].max() + all_vertices[:, 2].min()) * 0.5

    fig = plt.figure(figsize=(18, 12))

    angles = [(20, 30), (45, 45), (70, 60), (90, 90)]

    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        mesh = Poly3DCollection(all_face_vertices, alpha=0.1, edgecolor='k', linewidths=0.1)
        ax.add_collection3d(mesh)

        ax.set_title(f'View Angle (Elev: {elev}, Azim: {azim})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()
