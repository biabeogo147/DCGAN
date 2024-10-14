from pprint import pprint

from utils import *
from model import Full3DModel, FullLoss, PerceptualLoss


def visualize_pred_img(pred_img):
    # Convert tensor to NumPy array and transpose to (H, W, C)
    pred_img_np = pred_img.squeeze().cpu().numpy().transpose(2, 0, 1)

    # Display the image
    plt.imshow(pred_img_np)
    plt.axis('off')  # Turn off axis
    plt.show()


def inference():
    full_model = Full3DModel(image_size=256, device=device).to(device)  # Ensure model is on the correct device
    loss_fn = FullLoss(perceptual_loss=PerceptualLoss()).to(device)  # Ensure loss function is on the correct device

    # Example data
    images = torch.randn(1, 3, 240, 240).to(device)  # Move input to the correct device
    faces = torch.randint(0, 1000, (1, 500, 3)).to(device)  # Move input to the correct device
    real_img = torch.randn(1, 3, 256, 256).to(device)  # Move input to the correct device

    full_model.eval()
    with torch.no_grad():  # Ensure no gradients are calculated during inference
        pred_img = full_model(images, faces)

    visualize_pred_img(pred_img)
    pprint(pred_img)


if __name__ == "__main__":
    inference()
