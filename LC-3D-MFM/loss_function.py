from utils import *


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
    def __init__(self,
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