from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from model import Discriminator, Generator, Z_DIM, init_weights

EPOCHS = 300
BATCH = 64
LR = 2e-4
EXPORT_EVERY = 5

dev = "cuda" if torch.cuda.is_available() else "cpu"
Path("samples").mkdir(exist_ok=True)

data = (
    torch.from_numpy(np.load("data/fish64.npy"))
    .permute(0, 3, 1, 2)
    .float()
    .div(127.5)
    .sub(1.0)
    .to(dev)
)
N = data.size(0)

G = Generator().to(dev)
D = Discriminator().to(dev)
G.apply(init_weights)
D.apply(init_weights)

optG = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
fixed = torch.randn(64, Z_DIM, device=dev)


def export(path):
    was_training = G.training
    G.eval()
    out = {"fc": G.fc.weight.detach().cpu().float().numpy()}
    bns = [G.bn0, G.bn1, G.bn2, G.bn3]
    ups = [G.up1, G.up2, G.up3, G.up4]
    for i, bn in enumerate(bns):
        out[f"bn{i}_mean"] = bn.running_mean.detach().cpu().float().numpy()
        out[f"bn{i}_var"] = bn.running_var.detach().cpu().float().numpy()
        out[f"bn{i}_weight"] = bn.weight.detach().cpu().float().numpy()
        out[f"bn{i}_bias"] = bn.bias.detach().cpu().float().numpy()
    for i, up in enumerate(ups):
        out[f"up{i}"] = up.weight.detach().cpu().float().numpy()
    np.savez(path, **out)
    G.train(was_training)


for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(N, device=dev)
    d_sum = 0.0
    g_sum = 0.0
    steps = 0
    for i in range(0, N - BATCH + 1, BATCH):
        real = data[perm[i : i + BATCH]]
        flip = torch.rand(BATCH, device=dev) < 0.5
        real = torch.where(flip.view(-1, 1, 1, 1), real.flip(3), real)

        z = torch.randn(BATCH, Z_DIM, device=dev)
        fake = G(z)

        optD.zero_grad()
        d_loss = bce(D(real), torch.full((BATCH,), 0.9, device=dev)) + bce(
            D(fake.detach()), torch.zeros(BATCH, device=dev)
        )
        d_loss.backward()
        optD.step()

        optG.zero_grad()
        g_loss = bce(D(fake), torch.ones(BATCH, device=dev))
        g_loss.backward()
        optG.step()

        d_sum += d_loss.item()
        g_sum += g_loss.item()
        steps += 1

    print(f"epoch {epoch}/{EPOCHS}  D {d_sum / steps:.4f}  G {g_sum / steps:.4f}")

    if epoch % EXPORT_EVERY == 0 or epoch == EPOCHS:
        G.eval()
        with torch.no_grad():
            save_image(
                G(fixed),
                f"samples/epoch_{epoch:04d}.png",
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
            )
        G.train()
        export("generator.npz")
        torch.save({"G": G.state_dict(), "D": D.state_dict()}, "checkpoint.pt")
