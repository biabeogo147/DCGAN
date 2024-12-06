import os

import numpy as np


def count_vertices_and_faces(file_path):
    vertex, face = [[0, 0, 0]], []
    vertex_count, face_count = 0, 0
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    vertex_count += 1
                    vertex.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):
                    face_count += 1
                    face.append(list(map(int, line.strip().split()[1:])))
    else:
        print(f"File not found: {file_path}")
        return None, None

    return vertex, face, vertex_count, face_count


def load_obj(filename):
    vertices, faces, colors = [], [], []

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            if parts[0] == 'v':
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
            elif parts[0] == 'f':
                face = []
                for part in parts[1:]:
                    indices = part.split('/')
                    face.append(int(indices[0]) - 1)
                faces.append(face)
            elif parts[0] == 'usemtl':
                color = np.random.rand(3)
                colors.append(color)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    if not colors:
        colors = np.ones((faces.shape[0], 3), dtype=np.float32)
    else:
        colors = np.array(colors, dtype=np.float32)

    return vertices, faces, colors


if __name__ == "__main__":
    file_paths = [
        # "../../data/male.obj"
        # "D:/DS-AI/data/bfm2019_face_color.obj",
        # "D:/DS-AI/data/voxceleb3d/all.obj",
        # "D:/DS-AI/data/voxceleb3d/all_female.obj",
        # "D:/DS-AI/data/voxceleb3d/all_male.obj"
    ]
    vertex, face, vertex_count, face_count = count_vertices_and_faces(file_paths[0])
    print(vertex[:5])
    print(face[:5])
