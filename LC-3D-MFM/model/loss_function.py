import cv2
import torch
import numpy as np
from torch import nn
from torchvision import models
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation


class LandMarkLoss(nn.Module):
    def __init__(self):
        super(LandMarkLoss, self).__init__()

    def forward(self, geometry_pred, geometry_graph):
        """ We should Annotate 66 sparse 2D key points from [45] in paper """
        loss = F.mse_loss(geometry_pred, geometry_graph)
        return loss


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.lip_mask_predictor = segmentation.deeplabv3_resnet101(pretrained=True).eval()

    def forward(self, rendered_img, real_img):
        rendered_img = rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        real_img = real_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        mask1 = self.get_lip_mask(rendered_img)
        mask2 = self.get_lip_mask(real_img)

        distance_transform1 = cv2.distanceTransform(1 - mask1, cv2.DIST_L2, 5)
        distance_transform2 = cv2.distanceTransform(1 - mask2, cv2.DIST_L2, 5)

        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        loss1_to_2 = self.calculate_loss_from_contours(contours1, distance_transform2)
        loss2_to_1 = self.calculate_loss_from_contours(contours2, distance_transform1)
        total_loss = loss1_to_2 + loss2_to_1

        return total_loss

    def get_lip_mask(self, img):
        with torch.no_grad():
            output = self.lip_mask_predictor(img)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
        lip_mask = (output_predictions == 15).astype(np.uint8)
        return lip_mask

    def calculate_loss_from_contours(self, contours, distance_transform):
        total_points = 0
        loss = 0
        for contour in contours:
            total_points += len(contour)
            for point in contour:
                x, y = point[0]
                loss += distance_transform[int(y), int(x)]
        return loss / total_points if total_points > 0 else 0


class PhotometricLoss(nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()

    def forward(self, rendered_img, real_img):
        photometric_loss = 0
        for i in range(len(real_img)):
            Fi = real_img[i]
            Si = rendered_img[i]
            Mi = self.create_mask(Fi, Si)
            diff = Mi * (Fi - Si)
            photometric_loss += torch.sum(diff ** 2)

        return photometric_loss

    def create_mask(self, rendered_img, real_img, threshold=0.1):
        diff = torch.abs(real_img - rendered_img)
        diff_mean = torch.mean(diff, dim=-1, keepdim=True)
        mask = (diff_mean < threshold).float()
        return mask


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Perceptual Loss using VGG16
        vgg = models.vgg16(pretrained=True).features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.slice.parameters():
            param.requires_grad = False

    def forward(self, rendered_img, real_img):
        real_features = self.slice(real_img)
        rendered_features = self.slice(rendered_img)

        real_features = real_features.view(real_features.size(0), -1)
        rendered_features = rendered_features.view(rendered_features.size(0), -1)

        cosine_similarity = F.cosine_similarity(real_features, rendered_features, dim=-1)
        perceptual_loss = 1 - cosine_similarity.mean()

        return perceptual_loss


class GeometrySmoothnessLoss(nn.Module):
    def __init__(self):
        super(GeometrySmoothnessLoss, self).__init__()

    def forward(self, geometry_graph):
        smoothness_loss = 0.0
        batch_size, Ng, _ = geometry_graph.shape

        for i in range(1, Ng):
            node_position = geometry_graph[:, i, :]
            """ Where to get neighbor nodes? From topology of origin mesh?  """
            prev_node_position = geometry_graph[:, i - 1, :]
            smoothness_loss += F.mse_loss(node_position, prev_node_position)

        return smoothness_loss


class DisentanglementLoss(nn.Module):
    def __init__(self):
        super(DisentanglementLoss, self).__init__()

    def forward(self, expression_params):
        return torch.sum(expression_params ** 2)


class FullLoss(nn.Module):
    def __init__(self,
                 lambda_land=50.0, lambda_seg=0.001,
                 lambda_pho=2.5, lambda_per=1.0,
                 lambda_smo=10.0, lambda_dis=1.0):
        super(FullLoss, self).__init__()

        self.lambda_land = lambda_land
        self.lambda_seg = lambda_seg
        self.lambda_pho = lambda_pho
        self.lambda_per = lambda_per
        self.lambda_smo = lambda_smo
        self.lambda_dis = lambda_dis

        self.landmark_loss = LandMarkLoss()
        self.segmentation_loss = SegmentationLoss()
        self.photometric_consistency_loss = PhotometricLoss()
        self.perceptual_loss = PerceptualLoss()
        self.geometry_smoothness_loss = GeometrySmoothnessLoss()
        self.disentanglement_loss = DisentanglementLoss()

    def forward(self, rendered_img, real_img,
                geometry_pred, geometry_graph,
                expression_params):
        """ Normalize rendered_img before passing to FullLoss """
        Lland = self.landmark_loss(geometry_pred, geometry_graph)
        Lseg = self.segmentation_loss(rendered_img, real_img)
        Lpho = self.photometric_consistency_loss(rendered_img, real_img)
        Lper = self.perceptual_loss(rendered_img, real_img)
        Lsmo = self.geometry_smoothness_loss(geometry_pred)
        Ldis = self.disentanglement_loss(expression_params)
        total_loss = (self.lambda_land * Lland +
                      self.lambda_seg * Lseg +
                      self.lambda_pho * Lpho +
                      self.lambda_per * Lper +
                      self.lambda_smo * Lsmo +
                      self.lambda_dis * Ldis)
        return total_loss
