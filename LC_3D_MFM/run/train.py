from LC_3D_MFM.dataset_mfm.h5_analysis import get_face_properties_from_h5
from LC_3D_MFM.model import mini_modules, loss_function, image_formation
from torchvision.transforms import transforms
from LC_3D_MFM.dataset_mfm import voxceleb
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import torch


def train():
    lr = 0.001
    batch_size = 1
    num_workers = 20
    num_epochs = 1
    root_face_model = "D:/DS-AI/data/model2019_bfm.h5"
    root_data = "D:/DS-AI/data/voxceleb3d/Voxceleb3D_F-Z"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices, faces, colors = get_face_properties_from_h5(root_face_model)
    model = mini_modules.FullFaceModel(vertices, faces, colors).to(device)
    differential_render = image_formation.DifferentialRender()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = loss_function.LandMarkLoss()

    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
    ])

    train_dataset = voxceleb.VOXCeleb3dDataset(root_dir=root_data, transform=transform)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dataset:
            frames = torch.tensor(np.array([data['image'] for data in batch])).to(device)
            # vertices, triangles, poses = model(frames)

            # render_images = differential_render(vertices, triangles, poses)
            # loss = criterion(render_images, frames)
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            #
            # total_loss += loss.item()

    torch.save(model.state_dict(), 'pose_pretrained_model.pth')
    print("Pose pretraining completed and model saved.")


if __name__ == "__main__":
    train()
