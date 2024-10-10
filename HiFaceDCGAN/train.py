from model import Discriminator, HiFaceGenerator, weights_init
from utils import *


def visual_loss(G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def visual_g_progression(img_list):
    plt.axis("off")
    fig = plt.figure(figsize=(8, 8))
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())


def train():
    # Khởi tạo HiFace Generator và Discriminator
    generator = HiFaceGenerator(ngpu).to(device)
    discriminator = Discriminator(ngpu).to(device)

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Dataset và DataLoader (Ví dụ sử dụng CelebA hoặc dataset hình ảnh khuôn mặt)
    # dataset = datasets.CelebA(root='./data', transform=transform, download=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = data_loader()
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # Huấn luyện HiFace với DCGAN
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)

            # --- Huấn luyện Discriminator ---
            optimizer_d.zero_grad()

            # Phân biệt hình ảnh thật
            real_labels = torch.ones(batch_size, 1).to(device)
            dis_real = discriminator(real_images)
            real_loss = adversarial_loss(dis_real, real_labels)

            # Phân biệt hình ảnh giả từ HiFace
            z = torch.randn(batch_size, nz).to(device)
            fake_images = generator(z)  # HiFace tạo ảnh 3D
            fake_labels = torch.zeros(batch_size, 1).to(device)
            dis_fake = discriminator(fake_images.detach())
            fake_loss = adversarial_loss(dis_fake, fake_labels)

            # Tổng mất mát của Discriminator
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # --- Huấn luyện Generator (HiFace) ---
            optimizer_g.zero_grad()

            # Generator cố gắng đánh lừa Discriminator
            dis_fake = discriminator(fake_images)
            g_loss = adversarial_loss(dis_fake, real_labels)  # Muốn Discriminator nghĩ rằng ảnh giả là thật

            g_loss.backward()
            optimizer_g.step()

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            # In thông tin quá trình huấn luyện
            if i % 50 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] | Step [{i}/{len(dataloader)}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(z).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    visual_loss(G_losses, D_losses)
    visual_g_progression(img_list)


if __name__ == "__main__":
    train()
