from utils import *


class FaceParameterNet(nn.Module):
    def __init__(self, identity_dim=80, expression_dim=64, reflectance_dim=80, illumination_dim=9, pose_dim=6):
        super(FaceParameterNet, self).__init__()

        self.expression_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, expression_dim)
        )

        self.illumination_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, illumination_dim)
        )

        self.pose_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, pose_dim)
        )

        self.identity_reflectance_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, identity_dim + reflectance_dim)
        )

    def forward(self, per_frame_features, pooled_features, isFirst=False):
        expression = self.expression_net(per_frame_features)
        illumination = self.illumination_net(per_frame_features)
        pose = self.pose_net(per_frame_features)

        identity_reflectance = torch.zeros(1, pooled_features.shape[0])
        if isFirst:
            identity_reflectance = self.identity_reflectance_net(pooled_features)
        identity = identity_reflectance[:, :80]
        reflectance = identity_reflectance[:, 80:]

        return identity, reflectance, expression, illumination, pose


class SiameseModel(nn.Module):
    def __init__(self, identity_dim=80, expression_dim=64,
                 reflectance_dim=80, illumination_dim=9, pose_dim=6):
        super(SiameseModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 15 * 15, 512),
            nn.ReLU(),
        )

        self.face_param_net = FaceParameterNet(identity_dim, expression_dim,
                                               reflectance_dim, illumination_dim, pose_dim)

    def forward(self, frames):
        per_frame_features = []
        identities, reflectances, expressions, illuminations, poses = [], [], [], [], []

        for i in range(frames.shape[0]):
            x = frames[i]
            features = self.feature_extractor(x)
            per_frame_features.append(features)

        # (num_frames, feature_dim) -> (feature_dim)
        pooled_features = torch.mean(torch.stack(per_frame_features), dim=0)
        pooled_features = self.feature_extractor(pooled_features)

        for i in range(self.num_frames):
            identity, reflectance, expression, illumination, pose = self.face_param_net(per_frame_features[i],
                                                                                        pooled_features,
                                                                                        i == 0)

            expressions.append(expression)
            illuminations.append(illumination)
            poses.append(pose)

            if i == 0:
                identities.append(identity)
                reflectances.append(reflectance)

        return identities, reflectances, expressions, illuminations, poses


class FaceModel(nn.Module):
    def __init__(self, num_vertices=60000, num_graph_nodes=512, identity_dim=80, expression_dim=64, reflectance_dim=80):
        super(FaceModel, self).__init__()

        self.identity_model_graph = nn.Linear(identity_dim, num_graph_nodes * 3)
        self.expression_model_graph = nn.Linear(expression_dim, num_graph_nodes * 3)
        self.reflectance_model = nn.Linear(reflectance_dim, num_vertices * 3)

        # Fixed upsampling matrix for identity and expression
        """ We should precompute this before training """
        self.U = torch.randn(num_vertices * 3, num_graph_nodes * 3)

        """ We should use mean_face value from [4] in paper """
        self.mean_face = torch.zeros(1, num_vertices * 3)
        self.mean_face_reflectance = torch.zeros(1, num_vertices * 3)

    def forward(self, identity_params, expression_params, reflectance_params):
        identity_graph = self.identity_model_graph(identity_params)
        expression_graph = self.expression_model_graph(expression_params)

        # Upsampling
        identity_geometry = torch.matmul(self.U, identity_graph.T).T
        expression_geometry = torch.matmul(self.U, expression_graph.T).T

        face_geometry = self.mean_face + identity_geometry + expression_geometry
        reflectance = self.mean_face_reflectance + self.reflectance_model(reflectance_params)

        face_geometry = face_geometry.view(identity_params.shape[0], -1, 3)
        reflectance = reflectance.view(reflectance_params.shape[0], -1, 3)

        return face_geometry, reflectance


class FullFaceModel(nn.Module):
    def __init__(self, num_vertices=60000, num_graph_nodes=512,
                 identity_dim=80, expression_dim=64, reflectance_dim=80, illumination_dim=9, pose_dim=6):
        super(FullFaceModel, self).__init__()

        self.siamese_model = SiameseModel(identity_dim=identity_dim,
                                          expression_dim=expression_dim, reflectance_dim=reflectance_dim,
                                          illumination_dim=illumination_dim, pose_dim=pose_dim)

        self.face_model = FaceModel(num_vertices=num_vertices, num_graph_nodes=num_graph_nodes,
                                    identity_dim=identity_dim, expression_dim=expression_dim,
                                    reflectance_dim=reflectance_dim)

    def forward(self, frames):
        identities, reflectances, expressions, illuminations, poses = self.siamese_model(frames)
        face_geometry, reflectance_output = self.face_model(identities,
                                                            torch.mean(torch.stack(expressions), dim=0),
                                                            reflectances)
        """ face_geometry chỉ là tập toạ độ của các đỉnh trên khuôn mặt, 
         để tạo dựng một khuôn mặt hoàn chỉnh, chúng ta cần đỉnh trên mặt và tam giác khuôn mặt, 
         vì thế chúng ta cần cho các đỉnh đó đi qua mesh topology of Tewari để tạo thành bề mặt khuôn mặt liền mạch """
        return face_geometry, reflectance_output, illuminations, poses


if __name__ == "__main__":
    pass
