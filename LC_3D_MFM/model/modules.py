import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)  # Siamese = Yes
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1)  # Siamese = Yes
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1)  # Siamese = Yes
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=2)  # Siamese = Yes
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2)  # Siamese = Yes

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
        self.mean_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv1 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)  # Siamese = No
        self.conv2 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)  # Siamese = No
        self.fc1 = nn.Linear(256 * 4 * 4, 1000)  # Siamese = No
        self.fc2 = nn.Linear(1000, 1000)  # Siamese = No
        self.fc3 = nn.Linear(1000, 160)  # Siamese = No

    def forward(self, x):
        x = self.mean_pool(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        shape_param = x[:, :80]
        reflectance_param = x[:, 80:]
        return shape_param, reflectance_param


class ParameterEstimation(nn.Module):
    def __init__(self):
        super(ParameterEstimation, self).__init__()
        self.fc = nn.Linear(160, 14 * 14)  # Siamese = No
        self.conv1 = nn.Conv2d(1, 384, kernel_size=3, stride=1, padding=1)  # Siamese = No
        self.conv2 = nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1)  # Siamese = Yes
        self.conv3 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)  # Siamese = Yes
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)  # Siamese = Yes
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)  # Siamese = Yes
        self.fc1 = nn.Linear(6 * 6 * 256, 2048)  # Siamese = Yes
        self.fc2 = nn.Linear(2048, 6 + 64 + 27 + 1)  # Siamese = Yes

    def forward(self, shape_param, reflectance_param, low_features):
        x = F.relu(self.fc(torch.cat([shape_param, reflectance_param], dim=1)))
        x = x.view(-1, 1, 14, 14)
        x = F.relu(self.conv1(x))
        x = torch.cat([x, low_features], dim=1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        params = self.fc2(x)
        return params[:6, :], params[6:70, :], params[70:97, :], params[97]


class SiameseModel(nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.shared_identity_branch = SharedIdentity()
        self.parameter_estimation_branch = ParameterEstimation()

    def forward(self, x):
        medium_features, low_features = self.feature_extractor(x)
        shape_param, reflectance_param = self.shared_identity_branch(medium_features)

        poses, expressions, illuminations = [], [], []
        for i in range(low_features.shape[0]):
            pose, expression, illumination, _ = self.parameter_estimation_branch(shape_param, reflectance_param, low_features[i])
            poses.append(pose)
            expressions.append(expression)
            illuminations.append(illumination)

        return shape_param, reflectance_param, poses, expressions, illuminations


class FaceModel(nn.Module):
    def __init__(self, mean_face_path,
                 num_graph_nodes=512, identity_dim=80,
                 expression_dim=64, reflectance_dim=80):
        super(FaceModel, self).__init__()

        self.mean_face_path = mean_face_path
        self.mean_face, self.triangle_face = self.set_mean_face()
        num_vertices = len(self.mean_face)

        """ Find mean_face_reflectance """
        self.mean_face_reflectance = torch.zeros(1, num_vertices * 3)

        self.identity_model_graph = nn.Linear(identity_dim, num_graph_nodes * 3)
        self.expression_model_graph = nn.Linear(expression_dim, num_graph_nodes * 3)
        self.reflectance_model = nn.Linear(reflectance_dim, num_vertices * 3)

        # Fixed upsampling matrix for identity and expression
        """ We should precompute this before training """
        self.U = torch.randn(num_vertices * 3, num_graph_nodes * 3)


    def set_mean_face(self):
        mean_face = []
        triangle_face = []
        with open(self.mean_face_path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    mean_face.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):
                    triangle_face.append(list(map(int, line.strip().split()[1:])))

        return torch.tensor(mean_face), torch.tensor(triangle_face)


    def forward(self, identity_params, expression_params, reflectance_params):
        identity_graph = self.identity_model_graph(identity_params)
        expression_graph = self.expression_model_graph(expression_params)
        geometry_graph = (identity_graph + expression_graph).view(identity_params.shape[0], -1, 3)

        # Upsampling
        identity_geometry = torch.matmul(self.U, identity_graph.T).T.view(identity_params.shape[0], -1, 3)
        expression_geometry = torch.matmul(self.U, expression_graph.T).T.view(expression_params.shape[0], -1, 3)
        face_geometry = self.mean_face + identity_geometry + expression_geometry

        reflectance = (self.mean_face_reflectance
                       + self.reflectance_model(reflectance_params).view(reflectance_params.shape[0], -1, 3))

        return face_geometry, reflectance, geometry_graph


class FullFaceModel(nn.Module):
    def __init__(self, mean_face_path,
                 num_graph_nodes=512, identity_dim=80,
                 expression_dim=64, reflectance_dim=80):
        super(FullFaceModel, self).__init__()

        self.siamese_model = SiameseModel()
        self.face_model = FaceModel(mean_face_path=mean_face_path,
                                    num_graph_nodes=num_graph_nodes, identity_dim=identity_dim,
                                    expression_dim=expression_dim, reflectance_dim=reflectance_dim)

    def forward(self, frames):
        """ Normalizing images before feeding to the model """
        identity, reflectance, poses, expressions, illuminations = self.siamese_model(frames)
        face_geometry, reflectance_output, geometry_graph = self.face_model(identity.unsqueeze(0),
                                                            torch.mean(torch.stack(expressions), dim=0),
                                                            reflectance.unsqueeze(0))
        return face_geometry, reflectance_output, poses, geometry_graph, illuminations
