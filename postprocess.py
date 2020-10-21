import numpy as np


def _decode_boxes(raw_boxes, anchors):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    boxes = np.zeros(raw_boxes.shape, np.float32)
    SCALE = 128
    x_center = raw_boxes[..., 0] / SCALE * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / SCALE * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / SCALE * anchors[:, 2]
    h = raw_boxes[..., 3] / SCALE * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(6):
        offset = 4 + k * 2
        keypoint_x = raw_boxes[..., offset] / SCALE * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / SCALE * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes


def _tensors_to_detections(raw_box, raw_score, anchors):
    """The output of the neural network is a tensor of shape (b, 896, 16)
    containing the bounding box regressor predictions, as well as a tensor
    of shape (b, 896, 1) with the classification confidences.

    This function converts these two "raw" tensors into proper detections.
    Returns a list of (num_detections, 17) tensors, one for each image in
    the batch.

    This is based on the source code from:
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
    """
    detection_boxes = _decode_boxes(raw_box, anchors)
    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    mask = np.squeeze(raw_score) >= 1.  # (min_score_thresh)

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    boxes = detection_boxes[mask]
    scores = raw_score[mask]
    return np.concatenate((boxes, scores), axis=1)


def overlap_similarity(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inters = []
    area_a = ((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    for b in box_b:
        sx = max(b[0], box_a[0])
        sy = max(b[1], box_a[1])
        ex = min(b[2], box_a[2])
        ey = min(b[3], box_a[3])
        area_b = ((b[2] - b[0]) * (b[3] - b[1]))
        inter = (ex - sx) * (ey - sy)
        union = area_a + area_b - inter
        inters.append(inter / union)
    return np.array(inters)


def _weighted_non_max_suppression(detections):
    """The alternative NMS method as mentioned in the BlazeFace paper:

    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."

    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.

    The input detections should be a Tensor of shape (count, 17).

    Returns a list of PyTorch tensors, one for each detected face.

    This is based on the source code from:
    mediapipe/calculators/util/non_max_suppression_calculator.cc
    mediapipe/calculators/util/non_max_suppression_calculator.proto
    """
    if len(detections) == 0: return []

    output_detections = []

    # Sort the detections from highest to lowest score.
    remaining = np.argsort(detections[:, 16])[::-1]

    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        mask = ious > 0.3 #min_suppression_threshold
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection.copy()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :16]
            scores = detections[overlapping, 16:17]
            total_score = np.sum(scores)
            weighted = np.sum(coordinates * scores, axis=0) / total_score
            weighted_detection[:16] = weighted
            weighted_detection[16] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections