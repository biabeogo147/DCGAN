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
    generator = HiFaceGenerator().to(device).apply(weights_init)
    discriminator = Discriminator().to(device).apply(weights_init)

    if (device.type == 'cuda') and (ngpu > 1):
        generator = nn.DataParallel(generator, list(range(ngpu)))
        discriminator = nn.DataParallel(discriminator, list(range(ngpu)))

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    dataloader = data_loader()
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            # Train Discriminator
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Real images
            real_images = real_images.to(device)
            real_output = discriminator(real_images)
            real_loss = criterion(real_output, real_labels)

            # Fake images from HiFace
            noise = torch.randn(batch_size, 3, 64, 64).to(device)  # Input noise
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            fake_loss = criterion(fake_output, fake_labels)

            # Backpropagation for Discriminator
            d_loss = real_loss + fake_loss
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train HiFaceGenerator
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)

            # Backpropagation for Generator
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            if i % 50 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] | Step [{i}/{len(dataloader)}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    visual_loss(G_losses, D_losses)
    visual_g_progression(img_list)


if __name__ == "__main__":
    train()
