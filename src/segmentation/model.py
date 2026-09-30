import keras
from huggingface_hub import hf_hub_download


NUM_CLASSES = 104

HF_REPO_ID = "Sandeep-17/NutriVision-model"
MODEL_FILENAME = "nutrivision_segformer_b2.keras"


def build_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
    )

    model = keras.models.load_model(
        model_path,
        compile=False,
    )

    return model