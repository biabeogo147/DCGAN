from utils import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)


class HiFaceGenerator(nn.Module):
    def __init__(self):
        super(HiFaceGenerator, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Fully connected layers to regress coarse shape and identity coefficients
        self.fc_shape = nn.Linear(512 * 8 * 8, 512)
        self.fc_identity = nn.Linear(512, 256)  # Identity coefficient regression (β)
        self.fc_expression = nn.Linear(512, 128)  # Expression coefficient regression (ξ)
        self.fc_static_detail = nn.Linear(512, 300)  # Static detail coefficient (φ)
        self.fc_dynamic_detail = nn.Linear(128, 26)  # Dynamic detail coefficient (ϕ)

        # Reconstruct face model with displacement maps and vertex tension for 3D face
        self.fc_output_shape = nn.Linear(512, 3 * 64 * 64)  # Output coarse shape
        self.fc_albedo = nn.Linear(512, 3 * 64 * 64)  # Albedo map
        self.fc_sh = nn.Linear(512, 9)  # Spherical Harmonics coefficients for lighting

        # Interpolation layers for dynamic detail synthesis
        self.fc_compressed_basis = nn.Linear(26, 128)  # Compressed expression basis (B_com)
        self.fc_stretched_basis = nn.Linear(26, 128)  # Stretched expression basis (B_str)

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)

        # Coarse shape regression
        coarse_shape = torch.relu(self.fc_shape(x))
        coarse_shape_output = torch.tanh(self.fc_output_shape(coarse_shape)).view(x.size(0), 3, 64, 64)

        # Regress identity and expression coefficients
        identity_coeff = torch.relu(self.fc_identity(coarse_shape))  # β (identity)
        expression_coeff = torch.relu(self.fc_expression(coarse_shape))  # ξ (expression)

        # Regress static detail coefficients (φ)
        static_detail_coeff = torch.relu(self.fc_static_detail(coarse_shape))

        # Regress dynamic detail coefficients (ϕ)
        dynamic_detail_coeff = torch.relu(self.fc_dynamic_detail(expression_coeff))

        # Lighting via Spherical Harmonics (SH)
        sh_coeff = self.fc_sh(coarse_shape)

        # Interpolation for dynamic detail synthesis
        compressed_expression = torch.relu(self.fc_compressed_basis(dynamic_detail_coeff))
        stretched_expression = torch.relu(self.fc_stretched_basis(dynamic_detail_coeff))

        # Generate albedo map for skin texture
        albedo_map = torch.tanh(self.fc_albedo(coarse_shape)).view(x.size(0), 3, 64, 64)

        # Final 3D face output: Combine coarse shape, albedo, and lighting for rendering
        rendered_face = self.render_face(coarse_shape_output, albedo_map, sh_coeff)

        return rendered_face

    def render_face(self, coarse_shape, albedo_map, sh_coeff):
        # Function to combine the coarse shape, albedo, and lighting to produce the final rendered face
        # A simplified render using Spherical Harmonics and albedo

        # Simulated lighting (SH) applied to the albedo map
        lighting = sh_coeff.unsqueeze(-1).unsqueeze(-1)  # SH applied across all pixels
        shading = albedo_map * lighting  # Simulated shading
        rendered_output = coarse_shape + shading  # Combine coarse shape and shading

        return rendered_output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class HiFaceDCGAN(nn.Module):
    def __init__(self):
        super(HiFaceDCGAN, self).__init__()
        self.generator = HiFaceGenerator()  # HiFace as the Generator
        self.discriminator = Discriminator()  # A standard Discriminator from DCGAN

    def forward(self, x):
        generated_face = self.generator(x)
        real_or_fake = self.discriminator(generated_face)
        return generated_face, real_or_fake


if __name__ == '__main__':
    pass
