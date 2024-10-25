import os
from pprint import pprint

file_paths = [
    "D:/DS-AI/data/voxceleb3d/all.obj",
    # "D:/DS-AI/data/voxceleb3d/all_female.obj",
    # "D:/DS-AI/data/voxceleb3d/all_male.obj"
]

file_info = {}

for file_path in file_paths:
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = [file.readline().strip().split(' ') for _ in range(10)]
            file_info[file_path] = lines
    else:
        file_info[file_path] = "File not found."

pprint(file_info)
