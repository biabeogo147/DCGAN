from utils import *
from pytorch3d.renderer import *
from pytorch3d.structures import Meshes


class ImageFormation(nn.Module):
    def __init__(self, image_size=(256, 256), B=3):
        super(ImageFormation, self).__init__()
        self.image_size = image_size
        self.B = B  # Số lượng hệ số hài cầu

        # Thông số camera (có thể được học hoặc đặt bên ngoài)
        self.focal_length = nn.Parameter(torch.tensor([1000.0, 1000.0]))
        self.principal_point = nn.Parameter(torch.tensor([image_size[0]/2, image_size[1]/2]))

        # Thông số tư thế
        self.rotation = nn.Parameter(torch.zeros(3))  # Góc Euler
        self.translation = nn.Parameter(torch.zeros(3))

        # Thông số ánh sáng
        self.sh_coeffs = nn.Parameter(torch.randn(B**2))

        # Rasterizer có thể vi phân (đây chỉ là placeholder - triển khai thực tế sẽ phức tạp hơn)
        self.rasterizer = DifferentiableRenderer(image_size)

    def compute_rotation_matrix(self):
        # Chuyển đổi góc Euler thành ma trận xoay
        # Đây là phiên bản đơn giản hóa - bạn thường sẽ sử dụng một phương pháp mạnh mẽ hơn
        rx, ry, rz = self.rotation
        Rx = torch.tensor([[1, 0, 0],
                           [0, torch.cos(rx), -torch.sin(rx)],
                           [0, torch.sin(rx), torch.cos(rx)]])
        Ry = torch.tensor([[torch.cos(ry), 0, torch.sin(ry)],
                           [0, 1, 0],
                           [-torch.sin(ry), 0, torch.cos(ry)]])
        Rz = torch.tensor([[torch.cos(rz), -torch.sin(rz), 0],
                           [torch.sin(rz), torch.cos(rz), 0],
                           [0, 0, 1]])
        return Rz @ Ry @ Rx

    def apply_pose(self, vertices):
        R = self.compute_rotation_matrix()
        return vertices @ R.T + self.translation

    def project_3d_to_2d(self, vertices_3d):
        # Phép chiếu phối cảnh
        x, y, z = vertices_3d.unbind(-1)
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        x_proj = fx * x / z + cx
        y_proj = fy * y / z + cy
        return torch.stack([x_proj, y_proj], dim=-1)

    def compute_illumination(self, vertices, normals, reflectance):
        # Tính toán ánh sáng sử dụng hài cầu
        # Đây là phiên bản đơn giản hóa - triển khai thực tế sẽ liên quan đến các tính toán SH phức tạp hơn
        illumination = torch.zeros_like(vertices)
        for l in range(self.B**2):
            illumination += self.sh_coeffs[l] * self.evaluate_sh_basis(l, normals)
        return reflectance * illumination

    def evaluate_sh_basis(self, l, normals):
        # Placeholder cho việc đánh giá hàm cơ sở hài cầu
        # Triển khai thực tế sẽ tính toán các hàm cơ sở SH thích hợp
        return torch.ones_like(normals[..., 0])

    def forward(self, face_geometry, reflectance):
        # Áp dụng biến đổi tư thế
        posed_vertices = self.apply_pose(face_geometry)

        # Chiếu các đỉnh 3D thành 2D
        projected_vertices = self.project_3d_to_2d(posed_vertices)

        # Tính toán vector pháp tuyến của đỉnh (đơn giản hóa - tính toán thực tế sẽ phức tạp hơn)
        normals = nn.functional.normalize(posed_vertices, dim=-1)

        # Tính toán ánh sáng
        vertex_colors = self.compute_illumination(posed_vertices, normals, reflectance)

        # Render hình ảnh sử dụng rasterization có thể vi phân
        rendered_image = self.rasterizer(projected_vertices, vertex_colors)

        return rendered_image


class ImageFormation_2:
    def __init__(self, img_size=240):
        self.img_size = img_size

    def euler_to_rotation_matrix(self, alpha, beta, gamma):
        R_x = torch.tensor([[1, 0, 0],
                            [0, torch.cos(alpha), -torch.sin(alpha)],
                            [0, torch.sin(alpha), torch.cos(alpha)]])
        R_y = torch.tensor([[torch.cos(beta), 0, torch.sin(beta)],
                            [0, 1, 0],
                            [-torch.sin(beta), 0, torch.cos(beta)]])
        R_z = torch.tensor([[torch.cos(gamma), -torch.sin(gamma), 0],
                            [torch.sin(gamma), torch.cos(gamma), 0],
                            [0, 0, 1]])

        R = torch.mm(R_z, torch.mm(R_y, R_x))
        return R

    def apply_pose(self, face_geometry, pose):
        alpha, beta, gamma = pose[:, 0], pose[:, 1], pose[:, 2]
        translation_vector = pose[:, 3:6]

        rotation_matrix = self.euler_to_rotation_matrix(alpha, beta, gamma).to(face_geometry.device)

        face_geometry_transformed = torch.matmul(face_geometry, rotation_matrix.T)
        face_geometry_transformed += translation_vector[:, None, :]

        return face_geometry_transformed

    def perspective_projection(self, face_geometry_transformed, focal_length=800):
        projected_2d = face_geometry_transformed[:, :, :2] / (face_geometry_transformed[:, :, 2:] + 1e-5)

        projected_2d *= focal_length
        projected_2d += self.img_size // 2

        return projected_2d

    def apply_lighting(self, face_geometry_transformed, reflectance, illumination):
        batch_size = face_geometry_transformed.size(0)
        num_vertices = face_geometry_transformed.size(1)

        face_normals = torch.randn(batch_size, num_vertices, 3)
        face_normals = face_normals / (torch.norm(face_normals, dim=-1, keepdim=True) + 1e-5)

        # Ánh sáng: spherical harmonics (9 thành phần mỗi kênh màu RGB)
        sh_basis = self.compute_spherical_harmonics(face_normals)  # (batch_size, num_vertices, 9)

        # Tính toán ánh sáng bằng cách nhân hệ số spherical harmonics (illumination) với cơ sở spherical harmonics
        illumination_r = (illumination[:, :9].view(batch_size, 9, 1) * sh_basis).sum(dim=1)
        illumination_g = (illumination[:, 9:18].view(batch_size, 9, 1) * sh_basis).sum(dim=1)
        illumination_b = (illumination[:, 18:].view(batch_size, 9, 1) * sh_basis).sum(dim=1)

        # Kết quả ánh sáng tổng hợp cho các đỉnh
        illumination_combined = torch.stack([illumination_r, illumination_g, illumination_b], dim=-1)

        # Kết hợp với phản xạ da (reflectance) để tính màu sắc cuối cùng
        colors = reflectance * illumination_combined

        return colors

    # Hàm tính toán cơ sở spherical harmonics cho vector pháp tuyến
    def compute_spherical_harmonics(self, normals):
        """
        Tính toán cơ sở spherical harmonics bậc 2 cho vector pháp tuyến.
        """
        x = normals[:, :, 0]
        y = normals[:, :, 1]
        z = normals[:, :, 2]

        sh = torch.stack([
            torch.ones_like(x),  # L_0^0 (hằng số)
            y,  # L_1^-1
            z,  # L_1^0
            x,  # L_1^1
            x * y,  # L_2^-2
            y * z,  # L_2^-1
            3 * z ** 2 - 1,  # L_2^0
            x * z,  # L_2^1
            x ** 2 - y ** 2  # L_2^2
        ], dim=-1)  # (batch_size, num_vertices, 9)

        return sh

    # Hàm kết xuất ảnh 2D
    def render_image(self, face_geometry, reflectance, illumination, pose):
        """
        Kết xuất ảnh từ hình học khuôn mặt, phản xạ và ánh sáng.
        """
        # 1. Áp dụng phép biến đổi tư thế
        face_geometry_transformed = self.apply_pose(face_geometry, pose)

        # 2. Chiếu các đỉnh từ 3D sang 2D
        projected_2d = self.perspective_projection(face_geometry_transformed)

        # 3. Tính toán màu sắc sau khi áp dụng ánh sáng
        colors = self.apply_lighting(face_geometry_transformed, reflectance, illumination)

        # 4. Tạo hình ảnh từ các đỉnh được chiếu và màu sắc
        rendered_image = torch.zeros(face_geometry.size(0), 3, self.img_size, self.img_size)  # Tạo ảnh trống

        # Lặp qua từng điểm 2D để vẽ vào ảnh
        for i in range(face_geometry.size(1)):  # Duyệt qua từng đỉnh
            x_2d = projected_2d[:, i, 0].long()  # Tọa độ x
            y_2d = projected_2d[:, i, 1].long()  # Tọa độ y

            # Đảm bảo tọa độ nằm trong ảnh
            mask = (x_2d >= 0) & (x_2d < self.img_size) & (y_2d >= 0) & (y_2d < self.img_size)
            for b in range(face_geometry.size(0)):
                if mask[b]:
                    rendered_image[b, :, y_2d[b], x_2d[b]] = colors[b, i, :]  # Gán màu cho pixel

        return rendered_image


class DifferentiableRenderer(nn.Module):
    def __init__(self, image_size=256, device='cpu'):
        super(DifferentiableRenderer, self).__init__()

        self.device = device
        self.image_size = image_size
        self.cameras = PerspectiveCameras(device=self.device)
        self.lights = PointLights(location=[[0.0, 0.0, -3.0]], device=self.device)
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=HardPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        )

    def forward(self, vertices, faces, albedo):
        textures = TexturesVertex(verts_features=albedo)
        mesh = Meshes(verts=vertices, faces=faces, textures=textures).to(self.device)
        rendered_image = self.renderer(mesh)
        return rendered_image
