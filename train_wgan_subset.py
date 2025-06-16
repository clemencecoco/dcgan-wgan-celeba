import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.utils.data import DataLoader

# 数据路径
dataroot = "./img_align_celeba_subset"


# 参数
batch_size = 64
image_size = 64
nz = 100
ngf = 64
ndf = 64
nc = 3
lr = 1e-4
beta1 = 0.0
beta2 = 0.9
lambda_gp = 10
n_critic = 3
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("samples_wgan", exist_ok=True)

# 数据加载
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = dset.ImageFolder(root=dataroot, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 模型
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LayerNorm([ndf * 2, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LayerNorm([ndf * 4, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LayerNorm([ndf * 8, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.main(x).view(-1)

# 模型初始化
netG = Generator().to(device)
netD = Discriminator().to(device)
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

def compute_gp(D, real, fake):
    epsilon = torch.rand(real.size(0), 1, 1, 1, device=device)
    x_hat = epsilon * real + (1 - epsilon) * fake
    x_hat.requires_grad_(True)
    d_hat = D(x_hat)
    gradients = torch.autograd.grad(outputs=d_hat, inputs=x_hat,
                                    grad_outputs=torch.ones_like(d_hat),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

print("Starting WGAN-GP Training...")
for epoch in range(num_epochs):
    for i, (real, _) in enumerate(dataloader):
        real = real.to(device)
        b_size = real.size(0)

        # 训练 Discriminator
        for _ in range(n_critic):
            netD.zero_grad()
            z = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(z).detach()
            d_real = netD(real).mean()
            d_fake = netD(fake).mean()
            gp = compute_gp(netD, real, fake)
            loss_D = -d_real + d_fake + lambda_gp * gp
            loss_D.backward()
            optimizerD.step()

        # 训练 Generator
        netG.zero_grad()
        z = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(z)
        loss_G = -netD(fake).mean()
        loss_G.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                  f"Loss_D: {loss_D.item():.4f} "
                  f"Loss_G: {loss_G.item():.4f}")

    # 保存结果
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"samples_wgan1/fake_epoch_{epoch+1}.png", normalize=True)

print("WGAN-GP 训练完成！生成图保存在 samples_wgan/")
