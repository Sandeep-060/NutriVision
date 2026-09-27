import keras
import keras_cv

NUM_CLASSES = 103
IMAGE_SIZE = 640


def build_yolov8_detector(
    num_classes=NUM_CLASSES,
    learning_rate=1e-3,
    box_loss_weight=7.5,
    classification_loss_weight=2.0,
    backbone_trainable=True,
):
    """Build and compile a YOLOV8Detector.

    `classification_loss_weight` defaults to 2.0 (vs. KerasCV's library
    default of 0.5) for the reasons explained above. `box_loss_weight`
    is left at the library default since box regression was NOT the
    failing component in previous runs.
    """
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_s_backbone_coco"
    )
    backbone.trainable = backbone_trainable

    detector = keras_cv.models.YOLOV8Detector(
        num_classes=num_classes,
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=1,
    )

    detector.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            global_clipnorm=10.0,
        ),
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        box_loss_weight=box_loss_weight,
        classification_loss_weight=classification_loss_weight,
        jit_compile=False,
    )
    return detector


def set_backbone_trainable(detector, trainable):
    """Freeze/unfreeze the backbone in place. The detector must be
    re-`compile()`d (via `recompile_for_finetuning`) after calling this for
    the change to take effect on gradients — Keras bakes trainable state
    into the compiled train step."""
    detector.backbone.trainable = trainable
    return detector


def recompile_for_finetuning(
    detector,
    learning_rate=1e-4,
    box_loss_weight=7.5,
    classification_loss_weight=2.0,
):
    """Re-compile an already-built detector (e.g. after unfreezing the
    backbone) with a lower learning rate for phase-2 full fine-tuning."""
    detector.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            global_clipnorm=10.0,
        ),
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        box_loss_weight=box_loss_weight,
        classification_loss_weight=classification_loss_weight,
        jit_compile=False,
    )
    return detector