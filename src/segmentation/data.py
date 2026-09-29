from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


IMAGE_SIZE = (512, 512)
NUM_CLASSES = 104

EXCLUDED_IDS = {271, 2581, 3965, 4621}


def load_foodseg_record(record):
    image = Image.open(
        BytesIO(record["image"]["bytes"])
    ).convert("RGB")

    mask = Image.open(
        BytesIO(record["label"]["bytes"])
    )

    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)

    return image, mask


def preprocess(image, mask):
    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear"
    )

    mask = tf.image.resize(
        mask[..., None],
        IMAGE_SIZE,
        method="nearest"
    )

    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask[..., 0], tf.int32)

    return image, mask


def dataframe_to_dataset(df):
    def generator():
        for _, record in df.iterrows():

            record_id = int(record["id"])

            if record_id in EXCLUDED_IDS:
                continue

            image, mask = load_foodseg_record(record)

            yield preprocess(image, mask)

    output_signature = (
        tf.TensorSpec(
            shape=(512, 512, 3),
            dtype=tf.float32
        ),
        tf.TensorSpec(
            shape=(512, 512),
            dtype=tf.int32
        ),
    )

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )


def load_parquet_dataset(parquet_path):
    df = pd.read_parquet(parquet_path)
    return dataframe_to_dataset(df)