from typing import Dict

import torch
from PIL import Image
from neptune import Run
from omegaconf import DictConfig
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


class VisualizationLogger:
    def __init__(self, run: Run, config: DictConfig):
        self._run = run
        visualization_config = config.visualization
        self._n_images_per_epoch = visualization_config.n_images_per_epoch
        self._every = visualization_config.every

        self._n_logged = 0
        self._current_epoch = 0

    def log(self, epoch: int, image: torch.Tensor, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        if epoch != self._current_epoch:
            self._current_epoch = epoch
            self._n_logged = 0

        if epoch % self._every == 0 and self._n_logged < self._n_images_per_epoch:
            self._run["images"].append(
                visualize_object_detection(image, prediction, target),
                description=str(epoch)
            )
            self._n_logged += 1
