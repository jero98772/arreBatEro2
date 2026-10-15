import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
from PIL import Image

from qant_native_computing_toolkit import ai, utils

DEVICE_ID = 0
Z_DIM = 100
COUNT = 64
BATCH = 16
EPS = 1e-5
OUT = Path("generated")
OUT.mkdir(exist_ok=True)


def bf(x):
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32).astype(bfloat16))
    return utils.align_ndarray_page_boundary(arr)


w = np.load("generator.npz")
fc = bf(w["fc"])
bns = [
    (
        bf(w[f"bn{i}_mean"]),
        bf(w[f"bn{i}_var"]),
        bf(w[f"bn{i}_weight"]),
        bf(w[f"bn{i}_bias"]),
    )
    for i in range(4)
]
ups = [
    ai.upload_conv_transpose_weights(bf(w[f"up{i}"]), device_id=DEVICE_ID)
    for i in range(4)
]


def forward(z):
    x = ai.linear_fprop(bf(z), fc, device_id=DEVICE_ID)
    x = np.asarray(x, dtype=np.float32).reshape(-1, 512, 4, 4)
    for i in range(4):
        mean, var, gamma, beta = bns[i]
        x = ai.batchnorm2d_fprop(
            bf(x), mean, var, gamma, beta, EPS, device_id=DEVICE_ID
        )
        x = ai.relu_fprop(bf(np.asarray(x, dtype=np.float32)), device_id=DEVICE_ID)
        x = ai.conv_transpose_fprop_preloaded_weights(
            bf(np.asarray(x, dtype=np.float32)), ups[i], 1, 2, 1, 0
        )
    x = np.asarray(x, dtype=np.float32)
    x = ai.sigmoid_fprop(bf(2.0 * x), device_id=DEVICE_ID)
    x = np.asarray(x, dtype=np.float32) * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


rng = np.random.default_rng(int(time.time()))
batches = []
for _ in range(0, COUNT, BATCH):
    z = rng.standard_normal((BATCH, Z_DIM)).astype(np.float32)
    batches.append(forward(z))
images = np.concatenate(batches).transpose(0, 2, 3, 1)

stamp = int(time.time())
for i, img in enumerate(images):
    Image.fromarray(img).resize((256, 256), Image.BICUBIC).save(
        OUT / f"fish_{stamp}_{i:03d}.png"
    )

side = int(np.sqrt(len(images)))
grid = images[: side * side].reshape(side, side, 64, 64, 3)
grid = grid.transpose(0, 2, 1, 3, 4).reshape(side * 64, side * 64, 3)
Image.fromarray(grid).save(OUT / f"grid_{stamp}.png")
print("saved", len(images), "fish to", OUT)
