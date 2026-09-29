import numpy as np
import tensorflow as tf

from src.segmentation.postprocess import mask_to_boxes


def preprocess_image(image):
    image = tf.convert_to_tensor(image)

    image = tf.image.resize(
        image,
        (512, 512),
        method="bilinear",
    )

    image = tf.cast(image, tf.float32)

    # ImageNet normalization used by the finalized pipeline.
    image = image / 255.0
    image = (image - tf.constant(
        [0.485, 0.456, 0.406],
        dtype=tf.float32,
    )) / tf.constant(
        [0.229, 0.224, 0.225],
        dtype=tf.float32,
    )

    return image


def predict(model, image):
    original_height, original_width = image.shape[:2]

    processed = preprocess_image(image)
    batch = tf.expand_dims(processed, axis=0)

    logits = model(batch, training=False)

    predicted_mask = tf.argmax(
        logits,
        axis=-1,
    )[0]

    predicted_mask = tf.image.resize(
        predicted_mask[..., None],
        (original_height, original_width),
        method="nearest",
    )

    predicted_mask = tf.cast(
        predicted_mask[..., 0],
        tf.uint8,
    ).numpy()

    boxes = mask_to_boxes(
        predicted_mask,
        min_area=500,
    )

    return {
        "mask": predicted_mask,
        "boxes": boxes,
        "original_size": (
            original_width,
            original_height,
        ),
    }