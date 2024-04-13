
from torchmetrics.detection import MeanAveragePrecision


def mean_average_precision():
    # Must be used together with the postprocessor
    return MeanAveragePrecision(
        box_format="xyxy",
        average="macro",
        backend="faster_coco_eval"
    )
