from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

import keras_cv
from keras_cv import bounding_box

IMAGE_SIZE = 640
NUM_CLASSES = 103
MAX_BOXES_PER_IMAGE = 32  # observed max boxes/image in Stage 3 stats = 11


def _parse_yolo_label_file(label_path):
    """Read a YOLO-format label .txt file: `class_id cx cy w h` per line,
    all normalized to [0, 1]. Returns (classes (N,), boxes_xywh_norm (N,4))."""
    classes, boxes = [], []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_id, cx, cy, w, h = (float(x) for x in line.split())
            classes.append(cls_id)
            boxes.append([cx, cy, w, h])
    if not boxes:
        return np.zeros((0,), np.float32), np.zeros((0, 4), np.float32)
    return np.array(classes, np.float32), np.array(boxes, np.float32)


def _norm_xywh_to_pixel_xyxy(boxes_norm, img_w, img_h):
    """Normalized YOLO xywh -> absolute pixel xyxy, in ORIGINAL image pixel
    space (before any resize). This is the only manual box math in the
    pipeline; everything downstream of this is handled by KerasCV."""
    if boxes_norm.shape[0] == 0:
        return boxes_norm.reshape(0, 4).astype(np.float32)
    cx, cy, w, h = boxes_norm[:, 0], boxes_norm[:, 1], boxes_norm[:, 2], boxes_norm[:, 3]
    cx, w = cx * img_w, w * img_w
    cy, h = cy * img_h, h * img_h
    x1, y1 = cx - w / 2.0, cy - h / 2.0
    x2, y2 = cx + w / 2.0, cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def _list_pairs(images_dir, labels_dir):
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(images_dir.glob(ext))
    paths = sorted(paths)
    pairs = [(p, labels_dir / (p.stem + ".txt")) for p in paths]
    pairs = [(p, lp) for p, lp in pairs if lp.exists()]
    return pairs


def _make_generator(pairs):
    def gen():
        for img_path, label_path in pairs:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                image = np.asarray(im, dtype=np.float32)
            img_h, img_w = image.shape[0], image.shape[1]
            classes, boxes_norm = _parse_yolo_label_file(label_path)
            boxes_xyxy = _norm_xywh_to_pixel_xyxy(boxes_norm, img_w, img_h)
            yield {
                "images": image,
                "bounding_boxes": {"boxes": boxes_xyxy, "classes": classes},
            }

    return gen


def _build_raw_dataset(images_dir, labels_dir):
    pairs = _list_pairs(images_dir, labels_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No image/label pairs found under {images_dir} / {labels_dir}"
        )
    output_signature = {
        "images": tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        "bounding_boxes": {
            "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            "classes": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        },
    }
    ds = tf.data.Dataset.from_generator(
        _make_generator(pairs), output_signature=output_signature
    )
    return ds, len(pairs)


def build_dataset(
    images_dir,
    labels_dir,
    batch_size=4,
    shuffle=True,
    shuffle_buffer=1024,
    augment=False,
):
    """Build a `tf.data.Dataset` yielding `(images, bounding_boxes)` tuples,
    ready for `keras_cv.models.YOLOV8Detector.fit()`.

    Returns: (dataset, num_examples, steps_per_epoch)
    """
    ds, n = _build_raw_dataset(images_dir, labels_dir)

    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    # Bounding-box-aware, aspect-ratio-preserving letterbox resize.
    # Replaces ALL manual resize/pad/box-transform code from the previous
    # implementation — image and boxes are transformed together, correctly.
    resizing = keras_cv.layers.Resizing(
        IMAGE_SIZE,
        IMAGE_SIZE,
        pad_to_aspect_ratio=True,
        bounding_box_format="xyxy",
    )
    layers = [resizing]
    if augment:
        layers.append(
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy")
        )
    pipeline = keras_cv.layers.Augmenter(layers)

    ds = ds.ragged_batch(batch_size, drop_remainder=True)
    ds = ds.map(lambda sample: pipeline(sample), num_parallel_calls=tf.data.AUTOTUNE)

    def densify(sample):
        bboxes = bounding_box.to_dense(
            sample["bounding_boxes"], max_boxes=MAX_BOXES_PER_IMAGE
        )
        return sample["images"], bboxes

    ds = ds.map(densify, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = n // batch_size
    return ds, n, steps_per_epoch


def verify_batch(dataset):
    """Diagnostic helper: pull one batch and print shapes/dtypes/ranges so
    the pipeline can be sanity-checked BEFORE any GPU training starts."""
    images, bboxes = next(iter(dataset))
    print("images:", images.shape, images.dtype,
          "min/max:", float(tf.reduce_min(images)), float(tf.reduce_max(images)))
    print("boxes:", bboxes["boxes"].shape, bboxes["boxes"].dtype)
    print("classes:", bboxes["classes"].shape, bboxes["classes"].dtype)
    return images, bboxes