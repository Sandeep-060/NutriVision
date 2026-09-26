import keras
import keras_cv


NUM_CLASSES = 103


def build_yolov8_baseline(
    num_classes=NUM_CLASSES,
    learning_rate=1e-4,
):
    """
    Build the NutriVision YOLOv8-S baseline detector.
    """

    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_s_backbone_coco"
    )

    detector = keras_cv.models.YOLOV8Detector(
        num_classes=num_classes,
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=1,
    )

    detector.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        jit_compile=False,
    )

    return detector