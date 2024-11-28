from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, TexturesVertex, PointLights
)
from scipy.spatial.transform import Rotation as R
from pytorch3d.structures import Meshes
import torch.nn.functional as F
from torch import nn
import torch


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
        rotation = R.from_rotvec(pose).as_matrix()
        return torch.tensor(rotation, dtype=torch.float32, device=self.device)

    def forward(self, face_geometry, triangle_face, colors, pose):
        if not isinstance(face_geometry, torch.Tensor):
            face_geometry = torch.tensor(face_geometry, dtype=torch.float32, device=self.device)
        if not isinstance(triangle_face, torch.Tensor):
            triangle_face = torch.tensor(triangle_face, dtype=torch.int64, device=self.device)
        if not isinstance(colors, torch.Tensor):
            colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose, dtype=torch.float32, device=self.device)
        textures = TexturesVertex(verts_features=[colors])

        mesh = Meshes(verts=[face_geometry], faces=[triangle_face], textures=textures)
        center = face_geometry.mean(0)
        scale = (face_geometry - center).abs().max()
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))

        images = self.renderer(mesh, R=self.rotation_matrix(pose[:3]).unsqueeze(0), T=pose[3:].unsqueeze(0))
        return images[0, ..., :3]
