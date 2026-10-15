import hashlib
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

SIZE = 64
OUT = Path("data")
OUT.mkdir(exist_ok=True)

root = Path(
    snapshot_download(
        repo_id="imageomics/phylo-fish",
        repo_type="dataset",
        allow_patterns=["train/*", "test/*"],
    )
)

paths = sorted(
    p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

seen = set()
images = []
for p in paths:
    try:
        im = Image.open(p)
        im.load()
        im = im.convert("RGB")
    except Exception:
        continue
    im = im.resize((SIZE, SIZE), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8)
    if arr.std() < 8:
        continue
    digest = hashlib.md5(arr.tobytes()).hexdigest()
    if digest in seen:
        continue
    seen.add(digest)
    images.append(arr)

data = np.stack(images)
np.save(OUT / "fish64.npy", data)
print(len(paths), "found,", len(data), "kept ->", OUT / "fish64.npy")
