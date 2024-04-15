"""
Implementation of NMS from SAHI: https://github.com/obss/sahi
"""

import torch


def batched_nms(predictions: torch.tensor, match_metric: str = "IOU", match_threshold: float = 0.5):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    scores = predictions[:, 4].squeeze()
    category_ids = predictions[:, 5].squeeze()
    keep_mask = torch.zeros_like(category_ids, dtype=torch.bool)
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_indices = nms(predictions[curr_indices], match_metric, match_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    # sort selected indices by their scores
    keep_indices = keep_indices[scores[keep_indices].sort(descending=True)[1]].tolist()
    return keep_indices


def nms(
    predictions: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    # we extract coordinates for every
    # prediction box present in P
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]

    # we extract the confidence scores as well
    scores = predictions[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(idx.tolist())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[idx])
            # find the IoU of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # keep the boxes with IoU less than thresh_iou
        mask = match_metric_value < match_threshold
        order = order[mask]
    return keep
