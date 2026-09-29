from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "FoodSeg103"
PARQUET_DIR = RAW_ROOT / "data"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "foodseg103_segmentation"
)

EXCLUDED_IDS = {271, 2581, 3965, 4621}


def save_record(record, split, output_index):
    image = Image.open(
        BytesIO(record["image"]["bytes"])
    ).convert("RGB")

    mask = Image.open(
        BytesIO(record["label"]["bytes"])
    )

    image_array = np.asarray(image)
    mask_array = np.asarray(mask)

    # Skip the four known anomalous records.
    if int(record["id"]) in EXCLUDED_IDS:
        return None

    # Safety check.
    if image_array.shape[:2] != mask_array.shape:
        return None

    image_path = OUTPUT_ROOT / "images" / split / f"{output_index:05d}.jpg"
    mask_path = OUTPUT_ROOT / "masks" / split / f"{output_index:05d}.png"

    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image_array).save(image_path, quality=95)
    Image.fromarray(mask_array).save(mask_path)

    classes = sorted(
        int(x)
        for x in np.unique(mask_array)
        if int(x) != 0
    )

    return {
        "split": split,
        "original_id": int(record["id"]),
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
        "classes": ",".join(map(str, classes)),
        "num_classes": len(classes),
    }


def process_split(parquet_files, split):
    records = []
    output_index = 0

    for parquet_path in parquet_files:
        print(f"Processing {parquet_path.name}")

        df = pd.read_parquet(parquet_path)

        for _, record in df.iterrows():

            result = save_record(
                record,
                split,
                output_index,
            )

            if result is None:
                continue

            records.append(result)
            output_index += 1

    return records


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    train_files = sorted(
        PARQUET_DIR.glob("train-*.parquet")
    )

    val_files = sorted(
        PARQUET_DIR.glob("validation-*.parquet")
    )

    train_records = process_split(
        train_files,
        "train",
    )

    val_records = process_split(
        val_files,
        "val",
    )

    manifest = pd.DataFrame(
        train_records + val_records
    )

    manifest_path = OUTPUT_ROOT / "manifest.csv"
    manifest.to_csv(
        manifest_path,
        index=False,
    )

    class_counts = {}

    for _, row in manifest.iterrows():

        if not row["classes"]:
            continue

        for class_id in row["classes"].split(","):
            class_id = int(class_id)
            class_counts[class_id] = (
                class_counts.get(class_id, 0) + 1
            )

    statistics = pd.DataFrame(
        [
            {
                "foodseg103_class_id": class_id,
                "image_count": count,
            }
            for class_id, count
            in sorted(class_counts.items())
        ]
    )

    statistics_path = (
        OUTPUT_ROOT / "class_statistics.csv"
    )

    statistics.to_csv(
        statistics_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("STAGE 3 DATASET CREATION COMPLETE")
    print("=" * 60)

    print("Train images:", len(train_records))
    print("Validation images:", len(val_records))
    print("Total images:", len(manifest))
    print("Excluded IDs:", sorted(EXCLUDED_IDS))
    print("Manifest:", manifest_path)
    print("Class statistics:", statistics_path)


if __name__ == "__main__":
    main()