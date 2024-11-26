from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, TexturesVertex, PointLights
)
from scipy.spatial.transform import Rotation as R
from pytorch3d.structures import Meshes
import torch.nn.functional as F
from torch import nn
import torch


class ProjectFunction(nn.Module):
    def __init__(self, focal_length=800.0, image_size=240, device="cpu"):
        super(ProjectFunction, self).__init__()
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
        rotation = torch.tensor(R.from_rotvec(pose).as_matrix())
        return rotation

    def forward(self, face_geometry, triangle_face, colors, pose):
        if not isinstance(face_geometry, torch.Tensor):
            face_geometry = torch.tensor(face_geometry, dtype=torch.float32, device=self.device)
        if not isinstance(triangle_face, torch.Tensor):
            triangle_face = torch.tensor(triangle_face, dtype=torch.int64, device=self.device)
        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose, dtype=torch.float32, device=self.device)
        textures = TexturesVertex(verts_features=[colors])

        mesh = Meshes(verts=[face_geometry], faces=[triangle_face], textures=textures)
        center = face_geometry.mean(0)
        scale = max((face_geometry - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))

        images = self.renderer(mesh, R=self.rotation_matrix(pose[:3]).unsqueeze(0), T=pose[3:].unsqueeze(0))
        return images[0, ..., :3]


class DifferentiableRender(nn.Module):
    def __init__(self, image_size=240, device="cpu"):
        super(DifferentiableRender, self).__init__()
        self.device = device
        self.rotate_function = ProjectFunction()
        self.cameras = FoVPerspectiveCameras(device=self.device)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=RasterizationSettings(
                    image_size=image_size,
                    blur_radius=1e-6,
                    faces_per_pixel=50,
                )
            ),
            # Shader settings (will use Spherical Harmonics for illumination)
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras
            )
        )

    def forward(self, face_geometry, triangle_face, reflectance, pose, illumination):
        # face_geometry: (1, 60000, 3) tensor representing the vertices (x, y, z)
        # triangle_face: (1, ?, 3) tensor representing the vertex indices for each triangle face
        # reflectance: (1, 60000, 3) tensor representing the colors (R, G, B) for each vertex
        # pose: (1, 6) tensor representing rotation (3) and translation (3)
        # illumination: (1, 9) tensor representing spherical harmonics coefficients

        vertices = self.rotate_function(face_geometry, pose)

        # Interpolate reflectance attributes for each face
        """faces = self._create_faces(vertices)
        interpolated_reflectance = self.interpolate_vertex_attributes(vertices, faces, reflectance)
        textures = TexturesVertex(verts_features=interpolated_reflectance.view(batch_size, -1, 3))"""

        # Create texture for the mesh
        textures = TexturesVertex(verts_features=reflectance)

        # Create mesh
        """Cần áp dụng z-buffering để xác định visible triangle"""
        mesh = Meshes(verts=vertices, faces=triangle_face, textures=textures)

        # Apply illumination using spherical harmonics
        sh_illumination = self.apply_spherical_harmonics(vertices, reflectance, illumination)

        # Render the mesh with illumination
        images = self.renderer(mesh)
        return images * sh_illumination

    def apply_spherical_harmonics(self, vertices, reflectance, illumination):
        """
        Apply spherical harmonics illumination to the vertices.
        :param vertices: (1, num_vertices, 3) tensor of vertex positions
        :param reflectance: (1, num_vertices, 3) tensor of vertex colors
        :param illumination: (1, 9) tensor of spherical harmonics coefficients
        :return: (1, num_vertices, 3) tensor of illuminated colors
        """
        # Calculate normals for Lambertian reflection approximation
        normals = F.normalize(vertices, dim=-1)

        # Apply spherical harmonics to approximate lighting
        sh_basis = self.spherical_harmonics_basis(normals)
        lighting = torch.sum(sh_basis * illumination.unsqueeze(-1).unsqueeze(-1), dim=1)

        # Multiply by reflectance for Lambertian reflection
        illuminated_colors = reflectance * lighting.unsqueeze(-1)
        return illuminated_colors

    def spherical_harmonics_basis(self, normals):
        """
        Compute the spherical harmonics basis functions for given normals.
        :param normals: (batch_size, num_vertices, 3) tensor of vertex normals
        :return: (batch_size, 9, num_vertices) tensor of spherical harmonics basis values
        """
        x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
        sh_basis = torch.zeros(normals.shape[0], 9, normals.shape[1], device=self.device)
        sh_basis[:, 0, :] = 0.28209479177  # Y_00
        sh_basis[:, 1, :] = 0.4886025119 * y  # Y_1-1
        sh_basis[:, 2, :] = 0.4886025119 * z  # Y_10
        sh_basis[:, 3, :] = 0.4886025119 * x  # Y_11
        sh_basis[:, 4, :] = 1.09254843059 * x * y  # Y_2-2
        sh_basis[:, 5, :] = 1.09254843059 * y * z  # Y_2-1
        sh_basis[:, 6, :] = 0.31539156525 * (3 * z ** 2 - 1)  # Y_20
        sh_basis[:, 7, :] = 1.09254843059 * x * z  # Y_21
        sh_basis[:, 8, :] = 0.54627421529 * (x ** 2 - y ** 2)  # Y_22
        return sh_basis

    def interpolate_vertex_attributes(self, vertices, faces, attributes):
        """
        Interpolate vertex attributes using barycentric coordinates for each face.
        :param vertices: (batch_size, num_vertices, 3) tensor of vertex positions
        :param faces: (batch_size, num_faces, 3) tensor of vertex indices for each face
        :param attributes: (batch_size, num_vertices, attr_dim) tensor of vertex attributes (e.g., colors or normals)
        :return: (batch_size, num_faces, 3, attr_dim) interpolated attributes for each face
        """
        batch_size, num_faces, _ = faces.shape
        v0 = torch.gather(vertices, 1, faces[:, :, 0].unsqueeze(-1).expand(-1, -1, 3))
        v1 = torch.gather(vertices, 1, faces[:, :, 1].unsqueeze(-1).expand(-1, -1, 3))
        v2 = torch.gather(vertices, 1, faces[:, :, 2].unsqueeze(-1).expand(-1, -1, 3))

        # Compute vectors from v0 to v1 and v0 to v2
        v0v1 = v1 - v0
        v0v2 = v2 - v0

        # Compute the normal for each face using cross product
        face_normals = torch.cross(v0v1, v0v2, dim=2)
        face_normals = F.normalize(face_normals, dim=2)

        # Interpolate attributes using barycentric coordinates
        attr_v0 = torch.gather(attributes, 1, faces[:, :, 0].unsqueeze(-1).expand(-1, -1, attributes.shape[-1]))
        attr_v1 = torch.gather(attributes, 1, faces[:, :, 1].unsqueeze(-1).expand(-1, -1, attributes.shape[-1]))
        attr_v2 = torch.gather(attributes, 1, faces[:, :, 2].unsqueeze(-1).expand(-1, -1, attributes.shape[-1]))

        # Assuming uniform barycentric weights for simplicity (1/3 for each vertex)
        interpolated_attributes = (attr_v0 + attr_v1 + attr_v2) / 3.0

        return interpolated_attributes
