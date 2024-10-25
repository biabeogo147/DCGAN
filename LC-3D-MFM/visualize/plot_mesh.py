import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_obj_vertices(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
    return np.array(vertices)


def parse_obj_faces(file_path):
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('f '):
                parts = line.split()[1:]
                if len(parts) == 3:  # Checking if it's a triangular face
                    faces.append(parts)
    return faces


def get_face_vertices(vertices, faces):
    face_vertices = []
    for face in faces:
        indices = [int(idx.split('/')[0]) - 1 for idx in face]  # Extracting vertex indices (OBJ is 1-based index)
        face_vertices.append([vertices[i] for i in indices])
    return face_vertices


if __name__ == '__main__':
    file_path = "D:/DS-AI/data/voxceleb3d/all.obj"

    all_vertices = parse_obj_vertices(file_path)
    all_faces = parse_obj_faces(file_path)
    all_face_vertices = get_face_vertices(all_vertices, all_faces)

    max_range = np.array([all_vertices[:, 0].max() - all_vertices[:, 0].min(),
                          all_vertices[:, 1].max() - all_vertices[:, 1].min(),
                          all_vertices[:, 2].max() - all_vertices[:, 2].min()]).max() / 2.0

    mid_x = (all_vertices[:, 0].max() + all_vertices[:, 0].min()) * 0.5
    mid_y = (all_vertices[:, 1].max() + all_vertices[:, 1].min()) * 0.5
    mid_z = (all_vertices[:, 2].max() + all_vertices[:, 2].min()) * 0.5

    fig = plt.figure(figsize=(18, 12))

    # Different viewing angles
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
