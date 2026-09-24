from pathlib import Path
from collections import Counter

from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "foodseg103_yolo"
)

TRAIN_IMAGE_DIR = DATASET_DIR / "images" / "train"
VAL_IMAGE_DIR = DATASET_DIR / "images" / "val"

TRAIN_LABEL_DIR = DATASET_DIR / "labels" / "train"
VAL_LABEL_DIR = DATASET_DIR / "labels" / "val"

DATA_YAML = DATASET_DIR / "data.yaml"

EXPECTED_CLASSES = 103


# ============================================================
# HELPERS
# ============================================================

def validate_split(name, image_dir, label_dir):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    errors = 0

    image_files = sorted(image_dir.glob("*.jpg"))
    label_files = sorted(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    print("Images:", len(image_files))
    print("Labels:", len(label_files))

    # --------------------------------------------------------
    # Image / label count
    # --------------------------------------------------------

    if len(image_files) != len(label_files):
        print("ERROR: Image/label count mismatch")
        errors += 1
    else:
        print("Image/label counts: OK")

    # --------------------------------------------------------
    # Image / label pairing
    # --------------------------------------------------------

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    if missing_labels:
        print("Missing labels:", len(missing_labels))
        errors += len(missing_labels)

    if missing_images:
        print("Missing images:", len(missing_images))
        errors += len(missing_images)

    if not missing_labels and not missing_images:
        print("Image/label pairing: OK")

    # --------------------------------------------------------
    # Validate images
    # --------------------------------------------------------

    bad_images = 0

    for image_path in image_files:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            bad_images += 1
            print(
                f"BAD IMAGE: {image_path.name} -> {exc}"
            )

    if bad_images:
        print("Bad images:", bad_images)
        errors += bad_images
    else:
        print("Images readable: OK")

    # --------------------------------------------------------
    # Validate YOLO labels
    # --------------------------------------------------------

    annotation_count = 0
    class_counter = Counter()
    bad_labels = 0

    for label_path in label_files:

        try:
            text = label_path.read_text(encoding="utf-8").strip()

            if not text:
                print(f"EMPTY LABEL: {label_path.name}")
                bad_labels += 1
                continue

            for line_number, line in enumerate(
                text.splitlines(),
                start=1,
            ):

                parts = line.split()

                if len(parts) != 5:
                    print(
                        f"BAD FORMAT: {label_path.name} "
                        f"line {line_number}"
                    )
                    bad_labels += 1
                    continue

                try:
                    class_id = int(parts[0])

                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                except ValueError:
                    print(
                        f"BAD VALUES: {label_path.name} "
                        f"line {line_number}"
                    )
                    bad_labels += 1
                    continue

                # Class ID
                if not (0 <= class_id < EXPECTED_CLASSES):
                    print(
                        f"BAD CLASS ID: {label_path.name} "
                        f"line {line_number}: {class_id}"
                    )
                    bad_labels += 1
                    continue

                # Normalized coordinates
                if not (
                    0 <= x_center <= 1
                    and 0 <= y_center <= 1
                    and 0 < width <= 1
                    and 0 < height <= 1
                ):
                    print(
                        f"BAD COORDINATES: {label_path.name} "
                        f"line {line_number}"
                    )
                    bad_labels += 1
                    continue

                annotation_count += 1
                class_counter[class_id] += 1

        except Exception as exc:
            print(
                f"ERROR READING LABEL: "
                f"{label_path.name} -> {exc}"
            )
            bad_labels += 1

    print("Annotations:", annotation_count)

    if bad_labels:
        print("Invalid labels/annotations:", bad_labels)
        errors += bad_labels
    else:
        print("YOLO annotations: OK")

    # --------------------------------------------------------
    # Class coverage
    # --------------------------------------------------------

    missing_classes = [
        class_id
        for class_id in range(EXPECTED_CLASSES)
        if class_id not in class_counter
    ]

    if missing_classes:
        print(
            "Classes absent from this split:",
            len(missing_classes)
        )
        print(missing_classes)
    else:
        print("Class coverage: all 103 classes present")

    print("\nErrors:", errors)

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "annotations": annotation_count,
        "errors": errors,
        "class_counter": class_counter,
    }


# ============================================================
# DATA.YAML VALIDATION
# ============================================================

def validate_data_yaml():

    print("\n" + "=" * 70)
    print("DATA.YAML")
    print("=" * 70)

    if not DATA_YAML.exists():
        print("ERROR: data.yaml not found")
        return 1

    text = DATA_YAML.read_text(encoding="utf-8")

    required_entries = [
        "path: data/processed/foodseg103_yolo",
        "train: images/train",
        "val: images/val",
        "nc: 103",
        "  0: candy",
        "  102: other ingredients",
    ]

    errors = 0

    for entry in required_entries:
        if entry not in text:
            print(f"ERROR: Missing -> {entry}")
            errors += 1

    if errors == 0:
        print("data.yaml: OK")

    return errors


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FULL FOODSEG103 YOLO DATASET VALIDATION")
    print("=" * 70)

    required_paths = [
        TRAIN_IMAGE_DIR,
        VAL_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        VAL_LABEL_DIR,
        DATA_YAML,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        print("\nERROR: Required dataset paths are missing:")

        for path in missing_paths:
            print(path)

        raise SystemExit(1)

    train_stats = validate_split(
        "TRAIN",
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR,
    )

    val_stats = validate_split(
        "VALIDATION",
        VAL_IMAGE_DIR,
        VAL_LABEL_DIR,
    )

    yaml_errors = validate_data_yaml()

    total_errors = (
        train_stats["errors"]
        + val_stats["errors"]
        + yaml_errors
    )

    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Train images:       {train_stats['images']}"
    )
    print(
        f"Train labels:       {train_stats['labels']}"
    )
    print(
        f"Train annotations:  {train_stats['annotations']}"
    )

    print(
        f"Validation images:  {val_stats['images']}"
    )
    print(
        f"Validation labels:  {val_stats['labels']}"
    )
    print(
        f"Validation annotations: {val_stats['annotations']}"
    )

    print(
        f"\nTotal validation errors: {total_errors}"
    )

    if total_errors == 0:
        print("\nFULL DATASET VALIDATION PASSED")
    else:
        print("\nFULL DATASET VALIDATION FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()