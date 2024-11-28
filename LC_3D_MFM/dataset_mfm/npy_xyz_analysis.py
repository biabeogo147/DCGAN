import numpy as np
from matplotlib import pyplot as plt


def read_npy(file_path):
    try:
        data = np.load(file_path, allow_pickle=True).item()
        shape_data = data['shape']
        expr_data = data['expr']
        return shape_data, expr_data
    except FileNotFoundError:
        print(f"File {file_path} is not found.")
    except Exception as e:
        print(e)


def read_xyz(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = []
        for line in lines:
            values = line.split()
            data.append([float(values[0]), float(values[1]), float(values[2])])
        xyz_data = np.array(data)
        return xyz_data
    except FileNotFoundError:
        print(f"File {file_path} is not found.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    shape_data, expr_data = read_npy("D:/DS-AI/data/voxceleb3d/Voxceleb3D_F-Z/Faith_Hill/00000002.npy")
    xyz_data = read_xyz("D:/DS-AI/data/voxceleb3d/Voxceleb3D_F-Z/Faith_Hill/00000002.xyz")
    print(xyz_data.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz_data[:, 0], xyz_data[:, 1], xyz_data[:, 2], c='gray', marker='o')
    ax.view_init(elev=0, azim=0)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
