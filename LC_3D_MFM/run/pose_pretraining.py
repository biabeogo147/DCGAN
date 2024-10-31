from utils import *
from torch import optim
from .. model import modules
from .. dataset_mfm import voxceleb
from .. model import loss_function
from .. model import image_formation
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def pose_pretrain():
    model = modules.FullFaceModel().to(device)

    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = voxceleb.VOXCeleb3dDataset(root_dir="D:/DS-AI/data/voxceleb3d/Voxceleb3D_F-Z", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    land_mark_loss = loss_function.LandMarkLoss()

    rotate_function = image_formation.ProjectFunction()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for frames in train_loader:
            _, _, poses, geometry_graph, _ = model(frames)

            output = rotate_function(geometry_graph, poses)
            loss = land_mark_loss(output, geometry_graph)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Pose Pretraining Epoch [{epoch + 1}/{num_epochs // 3}], Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'pose_pretrained_model.pth')
    print("Pose pretraining completed and model saved.")


if __name__ == "__main__":
    pose_pretrain()
