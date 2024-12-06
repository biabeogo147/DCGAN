import torch
import torch.nn as nn
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    SoftPhongShader, TexturesVertex, FoVPerspectiveCameras, PointLights
)
from pytorch3d.structures import Meshes


class DifferentialRender(nn.Module):
    def __init__(self, focal_length=800.0, image_size=240, device="cpu"):
        super(DifferentialRender, self).__init__()
        self.device = device
        self.image_size = image_size
        self.focal_length = focal_length

        self.cameras = FoVPerspectiveCameras(device=self.device)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=RasterizationSettings(
                    image_size=image_size,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                )
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=PointLights(
                    device=self.device,
                    location=[[-1.0, -1.0, 3.0]],
                ),
            )
        )

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

        R = R_z @ (R_y @ R_x)
        return R

    def forward(self, face_geometry, triangle_face, colors, poses):
        face_geometry, triangle_face, colors, poses = face_geometry.to(self.device), triangle_face.to(
            self.device), colors.to(self.device), poses.to(self.device)

        textures = TexturesVertex(verts_features=[colors])

        mesh = Meshes(verts=[face_geometry], faces=[triangle_face], textures=textures)

        center = face_geometry.mean(0)
        scale = (face_geometry - center).abs().max()
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))

        images = [self.renderer(mesh, R=self.rotation_matrix(pose[:3]).unsqueeze(0), T=pose[3:].unsqueeze(0)) for pose
                  in poses]

        return torch.stack(images)
