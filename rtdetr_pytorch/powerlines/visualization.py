from typing import Dict

import torch
from PIL import Image
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes


POLES_COLOR = (223, 255, 0)


def inference_image_to_uint8(image: torch.Tensor) -> torch.Tensor:
    return image.squeeze().type(torch.uint8)


def visualize_object_detection(
    image: torch.Tensor, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], scale: float = 0.25
) -> Image:
    image_uint8 = inference_image_to_uint8(image)
    pred_visualization = draw_bounding_boxes(image_uint8, prediction["boxes"], colors=POLES_COLOR, width=3)
    target_visualization = draw_bounding_boxes(image_uint8, target["boxes"], colors=POLES_COLOR, width=3)
    visualization = torch.concatenate((pred_visualization, target_visualization), dim=2)

    height, width = visualization.shape[-2:]
    return to_pil_image((visualization / 255).float(), mode="RGB").resize((int(width * scale), int(height * scale)))
