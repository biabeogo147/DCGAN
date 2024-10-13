from utils import *
from pytorch3d.renderer import *
from pytorch3d.structures import Meshes


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Convolutional layers for feature extraction, with kernel size = 3
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Output: 64x120x120
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128x60x60
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 256x30x30
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers to get final feature vector
        self.fc = nn.Sequential(
            nn.Linear(256 * 30 * 30, 1024),  # Flatten the conv output
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)  # Output: 256D feature vector
        )

    def forward(self, x):
        x = self.conv(x)  # Convolutional feature extraction
        x = x.view(x.size(0), -1)  # Flatten the output from conv layers
        x = self.fc(x)  # Fully connected layers for final feature vector
        return x


class MorphableFaceModel(nn.Module):
    def __init__(self, num_identity_basis=80, num_expression_basis=64, num_albedo_basis=80):
        super(MorphableFaceModel, self).__init__()

        self.num_identity_basis = num_identity_basis
        self.num_expression_basis = num_expression_basis

        # basis (Identity + Expression)
        self.identity_basis = nn.Parameter(torch.randn(num_identity_basis, 1000, 3))  # Identity basis (num_identity_basis, num_vertices, 3)
        self.expression_basis = nn.Parameter(torch.randn(num_expression_basis, 1000, 3))  # Expression basis (num_expression_basis, num_vertices, 3)

        # basis for albedo
        self.albedo_basis = nn.Parameter(torch.randn(num_albedo_basis, 1000, 3))  # Albedo basis (num_albedo_basis, num_vertices, 3)

        self.mean_face = nn.Parameter(torch.randn(1000, 3))  # (num_vertices, 3)
        self.mean_albedo = nn.Parameter(torch.rand(1000, 3))  # (num_vertices, 3)

    def forward(self, identity_params, expression_params, albedo_params):
        batch_size = identity_params.shape[0]

        # 1. Tính toán hình dạng khuôn mặt từ tham số identity và expression
        identity_shape = torch.matmul(identity_params, self.identity_basis.view(identity_params.size(1), -1)).view(batch_size, 1000, 3)
        expression_shape = torch.matmul(expression_params, self.expression_basis.view(expression_params.size(1), -1)).view(batch_size, 1000, 3)

        vertices = self.mean_face + identity_shape + expression_shape  # (batch_size, num_vertices, 3)

        albedo = torch.matmul(albedo_params, self.albedo_basis.view(self.num_albedo_basis, -1))  # (batch_size, 1000 * 3)
        albedo = albedo.view(batch_size, 1000, 3)  # (batch_size, num_vertices, 3)
        albedo = self.mean_albedo + albedo  # (batch_size, num_vertices, 3)

        return vertices, albedo


class DifferentiableRenderer(nn.Module):
    def __init__(self, image_size=256, device='cpu'):
        super(DifferentiableRenderer, self).__init__()

        self.device = device
        self.image_size = image_size

        # Camera settings
        self.cameras = PerspectiveCameras(device=device)

        # Lighting setup
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        # Rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Shader settings
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=HardPhongShader(device=device, cameras=self.cameras, lights=self.lights)
        )

    def forward(self, vertices, faces, albedo):
        # Textures (albedo) setup for the mesh
        textures = TexturesVertex(verts_features=albedo)

        # Create the mesh structure using PyTorch3D's Meshes
        mesh = Meshes(verts=vertices, faces=faces, textures=textures)

        # Render the mesh to a 2D image
        rendered_image = self.renderer(mesh)
        return rendered_image


class Full3DModel(nn.Module):
    def __init__(self, image_size=256, device='cpu'):
        super(Full3DModel, self).__init__()
        self.siamese = SiameseNetwork()
        self.face_model = MorphableFaceModel()
        self.renderer = DifferentiableRenderer(image_size=image_size, device=device)

    def forward(self, images, faces):
        # 1. Feature extraction with Siamese Network
        features = self.siamese(images)
        identity_params = features[:, :80]  # First 80 dimensions for identity
        expression_params = features[:, 80:144]  # Next 64 dimensions for expression
        albedo_params = features[:, 144:224]  # Next 80 dimensions for albedo

        # 2. Create 3D Morphable Face Model
        vertices, albedo = self.face_model(identity_params, expression_params, albedo_params)

        # 3. Render the 3D model into a 2D image
        rendered_image = self.renderer(vertices, faces, albedo)

        return rendered_image


# ----- Perceptual Loss using VGG16 -----
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = models.vgg16(pretrained=True).features
        self.selected_layers = nn.Sequential(*list(vgg.children())[:23])

        # Freeze the parameters of VGG16 -> No Train
        for param in self.selected_layers.parameters():
            param.requires_grad = False

    def forward(self, input_img, target_img):
        input_features = self.selected_layers(input_img)
        target_features = self.selected_layers(target_img)
        perceptual_loss = F.mse_loss(input_features, target_features)
        return perceptual_loss


class FullLoss(nn.Module):
    def __init__(self, perceptual_loss,
                 lambda_land=1.0, lambda_seg=1.0,
                 lambda_pho=1.0, lambda_per=1.0,
                 lambda_smo=0.1, lambda_dis=0.01):
        super(FullLoss, self).__init__()
        #  perceptual_loss (nn.Module): Hàm perceptual loss sử dụng VGG16.
        self.perceptual_loss = PerceptualLoss()
        self.lambda_land = lambda_land
        self.lambda_seg = lambda_seg
        self.lambda_pho = lambda_pho
        self.lambda_per = lambda_per
        self.lambda_smo = lambda_smo
        self.lambda_dis = lambda_dis

    def forward(self,
                landmarks_pred, landmarks_true,
                lip_seg_pred, lip_seg_true,
                rendered_img, real_img,
                geometry_pred, smoothness_graph,
                expression_params):

        # 1. Landmark Consistency Loss (Lland)
        Lland = F.mse_loss(landmarks_pred, landmarks_true)

        # 2. Segmentation Consistency Loss (Lseg)
        Lseg = F.mse_loss(lip_seg_pred, lip_seg_true)

        # 3. Photometric Loss (Lpho)
        Lpho = F.l1_loss(rendered_img, real_img)

        # 4. Perceptual Loss (Lper) using VGG16
        Lper = self.perceptual_loss(rendered_img, real_img)

        # 5. Geometry Smoothness Loss (Lsmo)
        Lsmo = self.geometry_smoothness_loss(geometry_pred, smoothness_graph)

        # 6. Disentanglement Loss (Ldis)
        Ldis = torch.mean(expression_params ** 2)

        # Tổng hợp các hàm mất mát với trọng số
        total_loss = (self.lambda_land * Lland +
                      self.lambda_seg * Lseg +
                      self.lambda_pho * Lpho +
                      self.lambda_per * Lper +
                      self.lambda_smo * Lsmo +
                      self.lambda_dis * Ldis)
        return total_loss

    def geometry_smoothness_loss(self, geometry_pred, smoothness_graph):
        smoothness_loss = 0.0
        for g in geometry_pred:
            for neighbor in smoothness_graph[g]:
                smoothness_loss += torch.mean((geometry_pred[g] - geometry_pred[neighbor]) ** 2)
        return smoothness_loss


if __name__ == '__main__':
    pass