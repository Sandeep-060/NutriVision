import numpy as np
from scipy import ndimage


def mask_to_boxes(
    mask,
    min_area=500,
):
    """
    Convert a predicted FoodSeg103 segmentation mask
    into connected-component bounding boxes.
    """

    mask = np.asarray(mask)

    boxes = []

    class_ids = np.unique(mask)

    for class_id in class_ids:

        class_id = int(class_id)

        if class_id == 0:
            continue

        binary_mask = mask == class_id

        labeled, num_components = ndimage.label(
            binary_mask
        )

        for component_id in range(
            1,
            num_components + 1
        ):

            ys, xs = np.where(
                labeled == component_id
            )

            if len(xs) < min_area:
                continue

            boxes.append(
                {
                    "foodseg103_class_id": class_id,
                    "box": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ],
                    "area": int(len(xs)),
                }
            )

    return boxes