import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding="same")
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        low_features = F.relu(self.conv3(x))
        x = F.relu(self.conv4(low_features))
        medium_features = F.relu(self.conv5(x))
        return medium_features, low_features


class SharedIdentity(nn.Module):
    def __init__(self):
        super(SharedIdentity, self).__init__()
        self.conv1 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 160)

    def forward(self, x):
        x = torch.mean(x, dim=0, keepdim=True)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        identity_param = x[:, :80]
        reflectance_param = x[:, 80:]
        return identity_param, reflectance_param


class ParameterEstimation(nn.Module):
    def __init__(self):
        super(ParameterEstimation, self).__init__()
        self.fc = nn.Linear(160, 14 * 14)
        self.conv1 = nn.Conv2d(1, 384, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(6 * 6 * 256, 2048)
        self.fc2 = nn.Linear(2048, 6)

    def forward(self, identity_param, reflectance_param, low_features):
        x = F.relu(self.fc(torch.cat([identity_param, reflectance_param], dim=1)))
        x = x.view(-1, 14, 14)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, low_features], dim=0)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        pose_param = self.fc2(x)
        return pose_param


class SiameseModel(nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.shared_identity_branch = SharedIdentity()
        self.parameter_estimation_branch = ParameterEstimation()

    def forward(self, x):
        medium_features, low_features = self.feature_extractor(x)
        identity_param, reflectance_param = self.shared_identity_branch(medium_features)

        poses = []
        for i in range(low_features.shape[0]):
            pose = self.parameter_estimation_branch(identity_param, reflectance_param, low_features[i])
            poses.append(pose)
        poses = torch.stack(poses)

        return identity_param, reflectance_param, poses


class FaceModel(nn.Module):
    def __init__(self, mean_face, triangle_face, mean_reflectance, identity_dim=80, reflectance_dim=80):
        super(FaceModel, self).__init__()

        self.mean_face = mean_face
        self.triangle_face = triangle_face
        self.mean_reflectance = mean_reflectance
        num_vertices = len(self.mean_face)

        self.identity_model = nn.Linear(identity_dim, num_vertices * 3)
        self.reflectance_model = nn.Linear(reflectance_dim, num_vertices * 3)

    def forward(self, identity_params, reflectance_params):
        identity = self.identity_model(identity_params).view(identity_params.shape[0], -1, 3)
        mean_face = self.mean_face.expand(identity_params.shape[0], -1, -1)
        face_geometry = mean_face + identity

        reflectance = self.reflectance_model(reflectance_params).view(reflectance_params.shape[0], -1, 3)
        mean_reflectance = self.mean_reflectance.expand(reflectance_params.shape[0], -1, -1)
        face_reflectance = mean_reflectance + reflectance

        return face_geometry, face_reflectance


class FullFaceModel(nn.Module):
    def __init__(self, mean_face, triangle_face, mean_reflectance, identity_dim=80, reflectance_dim=80):
        super(FullFaceModel, self).__init__()

        self.siamese_model = SiameseModel()
        self.face_model = FaceModel(mean_face=mean_face, triangle_face=triangle_face, mean_reflectance=mean_reflectance,
                                    identity_dim=identity_dim, reflectance_dim=reflectance_dim)

    def forward(self, frames):
        identity, reflectance, poses = self.siamese_model(frames)
        face_geometry, face_reflectance = self.face_model(identity.unsqueeze(0), reflectance.unsqueeze(0))
        return face_geometry, face_reflectance, poses
