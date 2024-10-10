from model import Discriminator, Generator, weights_init
from utils import *


def visual_loss(G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(train_progress_path, 'loss.png'))
    plt.show()


def visual_g_progression(img_list):
    for idx, img in enumerate(img_list):
        plt.axis("off")
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(img, (1, 2, 0)), animated=True)
        plt.savefig(os.path.join(train_progress_path, f'epoch_{idx}.png'))
        plt.close(fig)


def compare_real_fake(dataloader, img_list):
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.axis("off")
    plt.title("Fake Images")
    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


def train():
    netG = Generator(ngpu).to(device).apply(weights_init)
    netD = Discriminator(ngpu).to(device).apply(weights_init)

    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # print(netG)
    # print(netD)

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()

    dataloader = data_loader()
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Train Discriminator
            # Real images
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            label = torch.ones(b_size, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            D_x = output.mean().item()

            # Fake images from Generator
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)

            label = torch.zeros(b_size, dtype=torch.float, device=device)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            D_G_z1 = output.mean().item()

            # Backpropagation for Discriminator
            errD = errD_real + errD_fake
            optimizerD.zero_grad()
            errD.backward()
            optimizerD.step()

            # Train Generator
            label = torch.ones(b_size, dtype=torch.float, device=device)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            D_G_z2 = output.mean().item()

            # Backpropagation for Generator
            optimizerG.zero_grad()
            errG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 50 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    visual_loss(G_losses, D_losses)
    visual_g_progression(img_list)

    torch.save(netG.state_dict(), os.path.join(model_path, 'generator.pth'))
    torch.save(netD.state_dict(), os.path.join(model_path, 'discriminator.pth'))


if __name__ == "__main__":
    train()