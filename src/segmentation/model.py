import keras


NUM_CLASSES = 104
MODEL_PATH = "models/segmentation/nutrivision_segformer_b2.keras"


def build_model():
    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    return model