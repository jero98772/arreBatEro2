import torch
import torch.nn as nn

Z_DIM = 100


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(Z_DIM, 512 * 4 * 4, bias=False)
        self.bn0 = nn.BatchNorm2d(512)
        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.up4 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 4, 4)
        x = torch.relu(self.bn0(x))
        x = torch.relu(self.bn1(self.up1(x)))
        x = torch.relu(self.bn2(self.up2(x)))
        x = torch.relu(self.bn3(self.up3(x)))
        return torch.tanh(self.up4(x))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
