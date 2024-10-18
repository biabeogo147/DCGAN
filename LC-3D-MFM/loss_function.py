import cv2
import dlib
from utils import *
import urllib.request
import torchvision.transforms as T


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

        shape_predictor_path = os.path.join(model_path, "shape_predictor_68_face_landmarks.dat")
        if not os.path.exists(shape_predictor_path):
            print("Downloading shape predictor model...")
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            urllib.request.urlretrieve(url, os.path.join(model_path, "shape_predictor_68_face_landmarks.dat.bz2"))
            import bz2
            with (bz2.BZ2File(os.path.join(model_path, "shape_predictor_68_face_landmarks.dat.bz2"))
                  as f_in, open(shape_predictor_path, "wb") as f_out):
                f_out.write(f_in.read())
            os.remove(os.path.join(model_path, "shape_predictor_68_face_landmarks.dat.bz2"))


        self.perceptual_loss = PerceptualLoss()
        self.lambda_land = lambda_land
        self.lambda_seg = lambda_seg
        self.lambda_pho = lambda_pho
        self.lambda_per = lambda_per
        self.lambda_smo = lambda_smo
        self.lambda_dis = lambda_dis
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(shape_predictor_path)

    def forward(self, rendered_img, real_img,
                lip_seg_pred, lip_seg_true,
                geometry_pred, smoothness_graph,
                expression_params):

        Lland = F.mse_loss(self.get_landmark(rendered_img), self.get_landmark(real_img))
        Lseg = F.mse_loss(lip_seg_pred, lip_seg_true)
        Lpho = F.l1_loss(rendered_img, real_img)
        Lper = self.perceptual_loss(rendered_img, real_img)
        Lsmo = self.geometry_smoothness_loss(geometry_pred, smoothness_graph)
        Ldis = torch.mean(expression_params ** 2)
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

    def get_landmark(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = self.face_detector(gray)[0]
        landmarks = self.landmark_predictor(gray, face)
        landmarks_points = torch.tensor([[p.x, p.y] for p in landmarks.parts()], dtype=torch.float32)
        return landmarks_points
