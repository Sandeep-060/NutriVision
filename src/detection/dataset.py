from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image


IMAGE_SIZE = 640
NUM_CLASSES = 103


def load_detection_sample(image_path, label_dir, image_size=IMAGE_SIZE):
    """
    Load one image and its YOLO annotation file.

    Converts:
        YOLO normalized xywh
    into:
        pixel xyxy
    """

    image_path = Path(image_path)
    label_dir = Path(label_dir)
    label_path = label_dir / f"{image_path.stem}.txt"

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image = np.asarray(image, dtype=np.float32)

    boxes = []
    classes = []

    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if not values:
                continue

            class_id, x_center, y_center, width, height = map(
                float, values
            )

            x1 = (x_center - width / 2) * image_size
            y1 = (y_center - height / 2) * image_size
            x2 = (x_center + width / 2) * image_size
            y2 = (y_center + height / 2) * image_size

            boxes.append([x1, y1, x2, y2])
            classes.append(class_id)

    return (
        image,
        np.asarray(boxes, dtype=np.float32),
        np.asarray(classes, dtype=np.float32),
    )


def dataset_generator(image_paths, label_dir, image_size=IMAGE_SIZE, shuffle=False):
    """
    Generate samples in KerasCV-compatible bounding-box format.
    """

    paths = list(image_paths)

    if shuffle:
        rng = np.random.default_rng(42)
        rng.shuffle(paths)

    for image_path in paths:
        image, boxes, classes = load_detection_sample(
            image_path,
            label_dir,
            image_size=image_size,
        )

        yield {
            "images": image,
            "bounding_boxes": {
                "boxes": boxes,
                "classes": classes,
            },
        }


def create_detection_dataset(
    image_paths,
    label_dir,
    batch_size=4,
    image_size=IMAGE_SIZE,
    shuffle=False,
):
    """
    Create a TensorFlow detection dataset.
    """

    output_signature = {
        "images": tf.TensorSpec(
            shape=(image_size, image_size, 3),
            dtype=tf.float32,
        ),
        "bounding_boxes": {
            "boxes": tf.TensorSpec(
                shape=(None, 4),
                dtype=tf.float32,
            ),
            "classes": tf.TensorSpec(
                shape=(None,),
                dtype=tf.float32,
            ),
        },
    }

    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(
            image_paths,
            label_dir,
            image_size=image_size,
            shuffle=shuffle,
        ),
        output_signature=output_signature,
    )

    dataset = dataset.ragged_batch(
        batch_size,
        drop_remainder=True,
    )

    dataset = dataset.map(
        lambda sample: (
            sample["images"] / 255.0,
            sample["bounding_boxes"],
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset