import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


class DifferentialRender(nn.Module):
    def __init__(self, image_size=240, device="cpu"):
        super(DifferentialRender, self).__init__()
        self.image_size = image_size
        self.device = device

    def rotation_matrix(self, pose):
        roll, pitch, yaw = pose

        R_x = torch.tensor([[1, 0, 0],
                            [0, torch.cos(roll), -torch.sin(roll)],
                            [0, torch.sin(roll), torch.cos(roll)]], dtype=torch.float32, device=self.device)
        R_y = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                            [0, 1, 0],
                            [-torch.sin(pitch), 0, torch.cos(pitch)]], dtype=torch.float32, device=self.device)
        R_z = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                            [torch.sin(yaw), torch.cos(yaw), 0],
                            [0, 0, 1]], dtype=torch.float32, device=self.device)
        R = torch.matmul(R_z, torch.matmul(R_y, R_x))
        return R

    def apply_pose(self, vertices, pose):
        translation = pose[:3]
        rotation_angles = pose[3:]
        rotation_matrix = self.rotation_matrix(rotation_angles)
        transformed_vertices = torch.matmul(vertices, rotation_matrix.T) + translation
        return transformed_vertices

    def compute_lighting(self, vertices, faces, light_positions):
        normals = self.compute_normals(vertices, faces)

        light_intensities = []
        for light_pos in light_positions:
            light_dir = light_pos - vertices
            light_dir = light_dir / light_dir.norm(dim=1, keepdim=True)
            dot_product = torch.sum(normals * light_dir, dim=1)
            dot_product = torch.clamp(dot_product, min=0)
            light_intensities.append(dot_product)

        total_light = torch.stack(light_intensities).sum(dim=0)
        return total_light

    def compute_normals(self, vertices, faces):
        v1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        v2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]

        normals = torch.cross(v1, v2)
        normals = normals / normals.norm(dim=1, keepdim=True)
        return normals

    def forward(self, vertices, faces, colors, poses, light_positions):
        all_projected_vertices = []

        for pose in poses:
            transformed_vertices = self.apply_pose(vertices, pose)
            projected_vertices = transformed_vertices[:, :2]

            light_intensities = self.compute_lighting(transformed_vertices, faces, light_positions)

            shaded_colors = colors * light_intensities.unsqueeze(1).expand_as(colors)
            all_projected_vertices.append((projected_vertices, shaded_colors))

        fig, ax = plt.subplots()
        for projected_vertices, shaded_colors in all_projected_vertices:
            for i in range(faces.shape[0]):
                face = faces[i]
                polygon = projected_vertices[face]
                color = shaded_colors[i % len(shaded_colors)].detach().cpu().numpy()
                ax.fill(*zip(*polygon), color=color.tolist(), alpha=0.7)

        ax.set_title("2D Projection of 3D Object with Multiple Poses and Lighting")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        ax.grid(True)
        plt.show()

        return all_projected_vertices
