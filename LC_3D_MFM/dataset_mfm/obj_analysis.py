import os


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


if __name__ == "__main__":
    file_paths = [
        # "../../data/male.obj"
        "D:/DS-AI/data/bfm2019_face_color.obj",
        # "D:/DS-AI/data/voxceleb3d/all.obj",
        # "D:/DS-AI/data/voxceleb3d/all_female.obj",
        # "D:/DS-AI/data/voxceleb3d/all_male.obj"
    ]
    vertex, face, vertex_count, face_count = count_vertices_and_faces(file_paths[0])
    print(vertex[:5])
    print(face[:5])