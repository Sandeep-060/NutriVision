import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from src.segmentation.model import build_model
from src.segmentation.inference import predict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_path = next(
    (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "foodseg103_segmentation"
        / "images"
        / "train"
    ).glob("*.jpg")
)

image = np.asarray(
    Image.open(image_path).convert("RGB")
)

model = build_model()

result = predict(
    model,
    image,
)

print("Original image:", image.shape)
print("Predicted mask:", result["mask"].shape)
print("Boxes detected:", len(result["boxes"]))
print("Boxes:", result["boxes"][:10])