from pathlib import Path
import io
import json
import argparse
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "FoodSeg103"
DATA_DIR = RAW_DIR / "data"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "foodseg103_yolo"
)

TRAIN_IMAGE_DIR = OUTPUT_DIR / "images" / "train"
VAL_IMAGE_DIR = OUTPUT_DIR / "images" / "val"

TRAIN_LABEL_DIR = OUTPUT_DIR / "labels" / "train"
VAL_LABEL_DIR = OUTPUT_DIR / "labels" / "val"

ID2LABEL_PATH = RAW_DIR / "id2label.json"


# DATASET FILES
TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00003.parquet",
    DATA_DIR / "train-00001-of-00003.parquet",
    DATA_DIR / "train-00002-of-00003.parquet",
]

VAL_FILES = [
    DATA_DIR / "validation-00000-of-00001.parquet",
]


# CONFIGURATION
BACKGROUND_CLASS_ID = 0
MIN_BOX_AREA = 16


# DIRECTORY SETUP
def create_directories():
    directories = [
        TRAIN_IMAGE_DIR,
        VAL_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        VAL_LABEL_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# IMAGE / MASK DECODING
def decode_image(value):
    if isinstance(value, dict):
        image_bytes = value["bytes"]
    else:
        image_bytes = value

    return Image.open(io.BytesIO(image_bytes))


def decode_mask(value):
    if isinstance(value, dict):
        mask_bytes = value["bytes"]
    else:
        mask_bytes = value

    return np.array(
        Image.open(io.BytesIO(mask_bytes))
    )


# MASK → BOUNDING BOX
def mask_to_boxes(mask):
    boxes = []
    class_ids = np.unique(mask)
    for class_id in class_ids:

        class_id = int(class_id)

        if class_id == BACKGROUND_CLASS_ID:
            continue

        ys, xs = np.where(mask == class_id)

        if len(xs) == 0:
            continue

        x_min = int(xs.min())
        y_min = int(ys.min())
        x_max = int(xs.max())
        y_max = int(ys.max())

        box_width = x_max - x_min
        box_height = y_max - y_min

        if box_width <= 0 or box_height <= 0:
            continue

        if box_width * box_height < MIN_BOX_AREA:
            continue

        boxes.append(
            {
                "class_id": class_id,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
        )

    return boxes


# BOX → YOLO
def box_to_yolo(box, image_width, image_height):

    foodseg_class_id = box["class_id"]

    # FoodSeg103:
    # 0 = background
    # 1–103 = food classes
    
    # YOLO:
    # 0–102 = food classes

    yolo_class_id = foodseg_class_id - 1

    x_min = box["x_min"]
    y_min = box["y_min"]
    x_max = box["x_max"]
    y_max = box["y_max"]

    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height

    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    assert 0 <= yolo_class_id <= 102

    assert 0 <= x_center <= 1
    assert 0 <= y_center <= 1

    assert 0 < width <= 1
    assert 0 < height <= 1

    return (
        yolo_class_id,
        x_center,
        y_center,
        width,
        height,
    )


# PROCESS ONE SAMPLE
def process_sample(
    image_value,
    label_value,
    sample_id,
    image_dir,
    label_dir,
):

    image = decode_image(image_value).convert("RGB")

    mask = decode_mask(label_value)

    image_width, image_height = image.size

    mask_height, mask_width = mask.shape[:2]

    if (image_width, image_height) != (mask_width, mask_height):
        raise ValueError(
            f"Image/mask size mismatch: "
            f"image=(width={image_width}, height={image_height}), "
            f"mask=(width={mask_width}, height={mask_height})"
        )

    boxes = mask_to_boxes(mask)

    if not boxes:
        return False, 0

    image_name = f"{int(sample_id):05d}.jpg"
    label_name = f"{int(sample_id):05d}.txt"

    image_path = image_dir / image_name
    label_path = label_dir / label_name

    image.save(
        image_path,
        format="JPEG",
        quality=95,
    )

    yolo_lines = []

    for box in boxes:

        (
            class_id,
            x_center,
            y_center,
            width,
            height,
        ) = box_to_yolo(
            box,
            image_width,
            image_height,
        )

        yolo_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    label_path.write_text(
        "\n".join(yolo_lines),
        encoding="utf-8",
    )

    return True, len(yolo_lines)


# PROCESS PARQUET
def process_parquet_file(
    parquet_file,
    image_dir,
    label_dir,
    split_name,
    stats,
    limit,
):
    """
    Process one FoodSeg103 Parquet file using low-memory Arrow access.
    Important:
    - Uses small Arrow batches.
    - Does NOT use batch.to_pydict().
    - Processes one sample at a time.
    - Uses the existing process_sample() function.
    """

    print(f"Processing: {parquet_file.name}")

    parquet = pq.ParquetFile(parquet_file)

    # Small batch to keep memory usage low.
    batch_size = 2

    for batch in parquet.iter_batches(batch_size=batch_size):

        image_column = batch.column("image")
        label_column = batch.column("label")
        id_column = batch.column("id")

        for row_idx in range(batch.num_rows):

            if limit > 0 and stats["total_seen"] >= limit:
                return

            stats["total_seen"] += 1

            sample_number = stats["total_seen"]

            try:
                image_value = image_column[row_idx].as_py()
                label_value = label_column[row_idx].as_py()
                sample_id = id_column[row_idx].as_py()

                success, annotation_count = process_sample(
                    image_value,
                    label_value,
                    sample_id,
                    image_dir,
                    label_dir,
                )

                if success:
                    stats["images_processed"] += 1
                    stats["annotations"] += annotation_count
                else:
                    stats["images_skipped"] += 1

            except Exception as exc:
                stats["errors"] += 1

                print(
                    f"\nERROR in {split_name} sample {sample_number}:"
                )
                print(str(exc))

            if stats["total_seen"] % 100 == 0:
                print(
                    f"{split_name}: "
                    f"{stats['total_seen']} samples processed"
                )

            # Release references from this iteration.
            image_value = None
            label_value = None

        # Release Arrow column references before next batch.
        del image_column
        del label_column
        del id_column


# PROCESS SPLIT
def process_split(
    parquet_files,
    image_dir,
    label_dir,
    split_name,
    limit,
):

    stats = {
        "total_seen": 0,
        "images_processed": 0,
        "images_skipped": 0,
        "annotations": 0,
        "errors": 0,
    }

    for parquet_file in parquet_files:

        if limit > 0 and stats["total_seen"] >= limit:
            break

        process_parquet_file(
            parquet_file,
            image_dir,
            label_dir,
            split_name,
            stats,
            limit,
        )

    return stats


# DATA.YAML
def create_data_yaml(id2label):

    food_labels = [
        id2label[str(class_id)]
        for class_id in range(1, 104)
    ]

    yaml_lines = [
        "path: data/processed/foodseg103_yolo",
        "train: images/train",
        "val: images/val",
        "",
        "nc: 103",
        "names:",
    ]

    for yolo_id, name in enumerate(food_labels):
        yaml_lines.append(
            f"  {yolo_id}: {name}"
        )

    yaml_path = OUTPUT_DIR / "data.yaml"

    yaml_path.write_text(
        "\n".join(yaml_lines),
        encoding="utf-8",
    )


# MAIN
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum samples per split. "
            "Use 0 for the complete dataset."
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FOODSEG103 → YOLO DATASET CONVERSION")
    print("=" * 70)

    print("\nMode:")

    if args.limit > 0:
        print(f"TEST — {args.limit} samples per split")
    else:
        print("FULL DATASET")

    with open(
        ID2LABEL_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        id2label = json.load(file)

    print("\nTotal label IDs:", len(id2label))
    print("Food classes:", len(id2label) - 1)

    create_directories()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAIN")
    print("=" * 70)

    train_stats = process_split(
        TRAIN_FILES,
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        "TRAIN",
        args.limit,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    val_stats = process_split(
        VAL_FILES,
        VAL_IMAGE_DIR,
        VAL_LABEL_DIR,
        "VAL",
        args.limit,
    )

    # Create YAML even during test.
    create_data_yaml(id2label)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)

    print("\nTRAIN")
    print("Samples:", train_stats["total_seen"])
    print("Images:", train_stats["images_processed"])
    print("Skipped:", train_stats["images_skipped"])
    print("Annotations:", train_stats["annotations"])
    print("Errors:", train_stats["errors"])

    print("\nVALIDATION")
    print("Samples:", val_stats["total_seen"])
    print("Images:", val_stats["images_processed"])
    print("Skipped:", val_stats["images_skipped"])
    print("Annotations:", val_stats["annotations"])
    print("Errors:", val_stats["errors"])

    print("\nOutput:")
    print(OUTPUT_DIR)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()