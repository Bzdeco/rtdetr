"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

by lyuwenyu
"""
import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from src.core import register


MAX_COST = torch.inf


@register
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        n_predictions = len(out_bbox)

        # Filter out non-degenerate boxes
        non_degenerate_mask = non_degenerate_bboxes_mask(box_cxcywh_to_xyxy(out_bbox))
        out_bbox_valid = out_bbox[non_degenerate_mask]
        out_prob_valid = out_prob[non_degenerate_mask]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        n_targets = len(tgt_ids)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob_valid = out_prob_valid[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob_valid**self.gamma) * (-(1 - out_prob_valid + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob_valid)**self.gamma) * (-(out_prob_valid + 1e-8).log())
            cost_class_valid = pos_cost_class - neg_cost_class
        else:
            cost_class_valid = -out_prob_valid[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox_valid = torch.cdist(out_bbox_valid, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou_valid = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox_valid), box_cxcywh_to_xyxy(tgt_bbox))

        # Set costs for all bounding boxes
        cost_shape = (n_predictions, n_targets)
        cost_bbox = torch.full(cost_shape, fill_value=MAX_COST, device=cost_bbox_valid.device)
        cost_bbox[non_degenerate_mask, :] = cost_bbox_valid
        cost_class = torch.full(cost_shape, fill_value=MAX_COST, device=cost_class_valid.device)
        cost_class[non_degenerate_mask, :] = cost_class_valid
        cost_giou = torch.full(cost_shape, fill_value=MAX_COST, device=cost_giou_valid.device)
        cost_giou[non_degenerate_mask, :] = cost_giou_valid
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def non_degenerate_bboxes_mask(bboxes: torch.Tensor) -> torch.Tensor:
    ordered_coordinates = torch.all(bboxes[:, :2] < bboxes[:, 2:], dim=1)
    within_image = torch.logical_and(torch.all(bboxes >= 0.0, dim=1), torch.all(bboxes <= 1.0, dim=1))
    return torch.logical_and(ordered_coordinates, within_image)
