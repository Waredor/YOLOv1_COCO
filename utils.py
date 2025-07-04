import torch
from collections import Counter
from dataset import CUBSATDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def intersection_over_union(pred_bboxes, target_bboxes,
                            box_format="midpoint"):
    '''
    Compute Intersection over Union

    Parameters:
        pred_bboxes (tensor): Predicted bounding boxes (BATCH_SIZE, 4)
        target_bboxes (tensor): Target bounding boxes (BATCH_SIZE, 4)
        box_format (str): corners of midpoint
            corners: [x1,y1,x2,y2]
            midpoint: [x,y,w,h]

    Return:
        IoU (scalar tensor): for ALL examples (BATCH_SIZE, 1)
    '''
    if not torch.is_tensor(pred_bboxes):
        pred_bboxes = torch.tensor(pred_bboxes)

    if not torch.is_tensor(target_bboxes):
        target_bboxes = torch.tensor(target_bboxes)

    box1_x1 = 0
    box1_y1 = 0
    box1_x2 = 0
    box1_y2 = 0
    box2_x1 = 0
    box2_y1 = 0
    box2_x2 = 0
    box2_y2 = 0

    if box_format == "midpoint":
        box1_x1 = pred_bboxes[..., 0:1] - pred_bboxes[..., 2:3] / 2
        box1_y1 = pred_bboxes[..., 1:2] - pred_bboxes[..., 3:4] / 2
        box1_x2 = pred_bboxes[..., 0:1] + pred_bboxes[..., 2:3] / 2
        box1_y2 = pred_bboxes[..., 1:2] + pred_bboxes[..., 3:4] / 2
        box2_x1 = target_bboxes[..., 0:1] - target_bboxes[..., 2:3] / 2
        box2_y1 = target_bboxes[..., 1:2] - target_bboxes[..., 3:4] / 2
        box2_x2 = target_bboxes[..., 0:1] + target_bboxes[..., 2:3] / 2
        box2_y2 = target_bboxes[..., 1:2] + target_bboxes[..., 3:4] / 2

    elif box_format == 'corners':
        box1_x1 = pred_bboxes[..., 0:1]
        box1_y1 = pred_bboxes[..., 1:2]
        box1_x2 = pred_bboxes[..., 2:3]
        box1_y2 = pred_bboxes[..., 3:4]
        box2_x1 = target_bboxes[..., 0:1]
        box2_y1 = target_bboxes[..., 1:2]
        box2_x2 = target_bboxes[..., 2:3]
        box2_y2 = target_bboxes[..., 3:4]

    cross_x1 = torch.max(box1_x1, box2_x1)
    cross_y1 = torch.max(box1_y1, box2_y1)
    cross_x2 = torch.min(box1_x2, box2_x2)
    cross_y2 = torch.min(box1_y2, box2_y2)

    # For non-overlapping boxes, clamp to 0 so that IoU=0
    intersection = (cross_x2 - cross_x1).clamp(0) * (cross_y2 - cross_y1).clamp(0)
    union = (
            (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
            + (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
            - intersection
    )

    return intersection / (union + 1e-6)

def non_max_suppression(
        bboxes, iou_threshold, confidence_threshold, box_format="midpoint"
):
    '''
    Performs Non Max Suppression on the given list of bounding boxes

    Parameters:
        bboxes (list): list[(class_prediction, confidence, x, y, w, h)]
        iou_threshold (float): minimum IoU required for predicted bbox to be correct
        confidence_threshold (float): minimum confidence required for predicted bbox.
                                all bboxes below this confidence are removed in prior to nms
        box_format (str): "corners" or "midpoint"

    Return:
        list: A list of bboxes after NMS performed
    '''
    nms_boxes = []
    assert type(bboxes) == list

    # [1] Remove all bboxes with confidence < confidence_threshold
    bboxes = [box for box in bboxes if box[1] > confidence_threshold]

    # [2] Sort bboxes for confidence in descending order
    bboxes.sort(key=lambda x: x[1], reverse=True)

    # [3] Perform nms for "EACH" class
    while(bboxes):
        top_box = bboxes.pop(0)
        nms_boxes.append(top_box)

        # [3-1] Don't compare if different class
        # [3-2] Only "leave" boxes with iou < iou_threshold
        bboxes = [
            box for box in bboxes
            if box[0] != top_box[0]
               or intersection_over_union(
                torch.tensor(box),
                torch.tensor(top_box[2:]),
                box_format=box_format
            ) < iou_threshold
        ]

    return nms_boxes

def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint",
        num_classes=3
):
    '''
    Calculates mAP for given predicted boxes and true boxes

    Parameters:
        pred_boxes (list): list of bounding boxes
            - (image_idx, class, confidence, x, y, w, h)
        true_boxes (list): list of ground truth bounding boxes
        iou_threshold (float): minimum iou required for bbox to be correct
        box_format (str): "corners" or "midpoint"
        num_classes (int): number of classes

    Returns:
        mAP (float): mAP across "all" classes
    '''

    # [NOTE] The ultimate goal is to find TP & FP for each pred_boxes with class c

    # average precisions -> later will be averaged = mAP
    average_precisions = []

    for c in range(num_classes):
        # pred_boxes for current class c
        class_pred_boxes = [
            box for box in pred_boxes
            if box[1] == c
        ]

        # true boxes for current class c
        class_gt_boxes = [
            box for box in true_boxes
            if box[1] == c
        ]

        # If there's no gt box, skip
        if len(class_pred_boxes) == 0:
            continue

        # Build a frequency dictionary for each image index
        # This tells how many true boxes per image index
        gt_visited = Counter([
            gt[0] for gt in class_gt_boxes
        ])

        # convert value: num boxes -> [0] * num_boxes
        # to make visited array for each box
        for key, val in gt_visited.items():
            gt_visited[key] = torch.zeros(val)  # ТУТ НАДО РАЗОБРАТЬСЯ ПОДРОБНЕЕ, ЧТО ПРОИСХОДИТ

        # Time to calculate TP/FP
        #First, sort class_pred_boxes w.r.t. confidence
        class_pred_boxes.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(class_pred_boxes))
        FP = torch.zeros(len(class_pred_boxes))
        total_gt_boxes = len(class_gt_boxes)

        for detection_idx, detection in enumerate(class_pred_boxes):
            best_iou = 0
            best_iou_gt_idx = None

            # GT boxes for SAME image and SAME class
            same_image_class_gt_boxes = [
                box for box in class_gt_boxes
                if box[0] == detection[0]
            ]

            for gt_idx, gt in enumerate(same_image_class_gt_boxes):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_iou_gt_idx = gt_idx

            if best_iou > iou_threshold:
                # If not visited, then the current predicted detection is correct!
                if gt_visited[detection[0]][best_iou_gt_idx] == 0:
                    gt_visited[detection[0]][best_iou_gt_idx] = 1
                    TP[detection_idx] = 1
                else: # If already visited, then the current predicted detection is incorrect!
                    FP[detection_idx] = 1
            else: # If best iou < threshold, then the current pred detection failed!
                FP[detection_idx] = 1

        # Now, we found all TP, FP for current class
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        precisions = TP_cumsum / (FP_cumsum + TP_cumsum + 1e-6)
        recalls = TP_cumsum / (total_gt_boxes + 1e-6)

        precisions = torch.cat(
            (torch.tensor([1]), precisions)
        )
        recalls = torch.cat(
            (torch.tensor([0]), recalls)
        )

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def get_bboxes(
        loader, model,
        iou_threshold, confidence_threshold,
        box_format="midpoint",
        device="cpu",
        s=7
):
    '''
    Returns a tuple of list of all the bounding boxes information for both prediction
    boxes and ground truth boxes in shape (image idx, class, confidence, x, y, w, h)

    Parameters:
        loader (generator): DataLoader
        model (nn.Module): YOLOv1 model
        iou_threshold: min. iou required for predicted bbox to be correct
        confidence_threshold: min. confidence to be a candidate
        box_format (str): "corners" or "midpoint"
        device (str): "cpu" or "gpu"

    Returns:
        tuple: (all_pred_boxes, all_gt_boxes)
        - decoupled bounding box information accross all batches and examples
        for predictions and ground truths
    '''
    # We're not training
    model.eval()
    train_idx = 0

    # All boxes to return: (train_idx, class, confidence, x, y, w, h)
    all_pred_boxes = []
    all_gt_boxes = []

    # For each BATCH
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            predictions = model(images)

        print("prediction shape:", predictions.shape)

        # after forward(), its shape is (batch_size, s*s*(c+5*b))
        batch_size = images.shape[0]
        #predicitons = (batchsize, s, s, c+5*b)
            # each predictions[bathsize, ...] is predictions per ONE IMAGE
        # labels = (batchsize, s, s, c+5*b)

        pred_boxes = cellboxes_to_list_boxes(predictions)
        gt_boxes = cellboxes_to_list_boxes(labels)
        # pred_boxes = [list of (class, confidence, x, y, w, h)] * batch_size
        # labels = [list of (class, confidence, x, y, w, h)] * batch_size


        # for each IMAGE
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                pred_boxes[idx],
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold,
                box_format=box_format
            )
            print("nms_boxes:", nms_boxes)
            # For each PREDICTED BOX
            for nms_box in nms_boxes:
                # We need "train idx" for mAP
                all_pred_boxes.append([train_idx] + nms_box)

            # For each GROUND TRUTH BOX
            for gt_box in gt_boxes[idx]:
                # Many (i, j)th cells in S x S DO NOT have
                # ground truth labels!
                if gt_box[1] > confidence_threshold:
                    all_gt_boxes.append([train_idx] + gt_box)
            train_idx += 1

    model.train()
    return all_pred_boxes, all_gt_boxes

def convert_cellboxes(box_3d, s=7, b=2, c=3):
    '''
    Converts (batch_size, s, s, c + 5 * b) -> (batch_size, s, s, 6)
    by selecting the best box among box 1 and 2.

    Parameters:
        box_3d (torch.Tensor): Shape of (batch_size, s, s, c + 5 * b)
        s (int): Grid size

    Returns:
        tensor: Shape of (batch_size, s, s, 6) where
        tensor vector in dim=3 is in the format of
        (class, confidence, x, y, w, h)

    For 13 vector,
    0:3 = class vector
    3 = box1 confidence
    4:8 = box1 (x,y,w,h)
    8 = box2 confidence
    9:13 = box2 (x,y,w,h)
    '''
    batch_size = box_3d.shape[0]
    box_3d = box_3d.to("cpu")
    box_3d = box_3d.reshape(batch_size, s, s, 5 * b+ c)

    converted_boxes = torch.empty(batch_size, s, s, 6)
    for i in range(s):
        for j in range(s):
            # Scale to relative to the whole image rather than cell
            # THIS CODE IS FOR 3 CLASSES!!!
            box1_cell_x = box_3d[..., i:i + 1, j:j + 1, 4]
            box1_cell_y = box_3d[..., i:i + 1, j:j + 1, 5]
            box1_cell_w = box_3d[..., i:i + 1, j:j + 1, 6]
            box1_cell_h = box_3d[..., i:i + 1, j:j + 1, 7]

            box_3d[..., i:i + 1, j:j + 1, 4] = (j + box1_cell_x) / s
            box_3d[..., i:i + 1, j:j + 1, 5] = (i + box1_cell_y) / s
            box_3d[..., i:i + 1, j:j + 1, 6] = box1_cell_w / s
            box_3d[..., i:i + 1, j:j + 1, 7] = box1_cell_h / s

            box2_cell_x = box_3d[..., i:i + 1, j:j + 1, 9]
            box2_cell_y = box_3d[..., i:i + 1, j:j + 1, 10]
            box2_cell_w = box_3d[..., i:i + 1, j:j + 1, 11]
            box2_cell_h = box_3d[..., i:i + 1, j:j + 1, 12]

            box_3d[..., i:i + 1, j:j + 1, 9] = (j + box2_cell_x) / s
            box_3d[..., i:i + 1, j:j + 1, 10] = (i + box2_cell_y) / s
            box_3d[..., i:i + 1, j:j + 1, 11] = box2_cell_w / s
            box_3d[..., i:i + 1, j:j + 1, 12] = box2_cell_h / s

            # Best Class
            best_class = box_3d[..., i:i + 1, j:j + 1, 0:3].argmax(dim=3, keepdim=True)

            # Best confidence
            # Best confidence idx for identity
            best_confidence, best_confidence_idx = torch.cat(
                (
                    box_3d[..., i:i + 1, j:j + 1, 3:4].argmax(dim=3, keepdim=True),
                    box_3d[..., i:i + 1, j:j + 1, 8:9].argmax(dim=3, keepdim=True)
                ),
                dim=3
            ).max(dim=3, keepdim=True)

            boxes1 = box_3d[..., i:i + 1, j:j + 1, 3:8]
            boxes2 = box_3d[..., i:i + 1, j:j + 1, 9:12]

            # Best Box coordinate
            best_box = (
                    (1 - best_confidence_idx) * boxes1
                    + best_confidence_idx * boxes2
            )

            converted_boxes[..., i:i + 1, j:j + 1, :] = torch.cat(
                (
                    best_class,
                    best_confidence,
                    best_box,
                ),
                dim=-1
            )

    return converted_boxes

# input: (128, 7, 7, 30)
# return [[[class,confidence,x,y,w,h],[],[]],[],...,[]], each element -> all
# boxes per image: 3D
def cellboxes_to_list_boxes(box_3d, s=7):
    '''
    Returns a list of "list of all bounding box information" in the format of
    (class, confidence, x, y, w, h). The length of the output will be the same
    as batch_size. Each ELEMENT of output is also a LIST for a particular IMAGE.

    Parameters:
        box_3d (torch.Tensor): Shape of (batch_size, s, s, 13). Each element in dim=0
        is a 3D-shaped bounding boxes.

        S (int): Grid size

    Returns:
        list: The length of output will be the same as batch_size.
        Each ELEMENT of output is also a 2D LIST for a particular IMAGE.
        Each bounding box is (class, confidence, x, y, w, h)
    '''
    converted_list_boxes = []
    batch_size = box_3d.shape[0]

    # convert (batch_size,s,s,13) -> (batch_size,s,s,6)
    box_3d = convert_cellboxes(box_3d)
    print("box_3d in cellboxes_to_list_boxes:", box_3d.shape)
    box_3d[..., 0] = box_3d[..., 0].long()

    for img_idx in range(batch_size):
        img_list_boxes = []
        for box_idx in range(s*s):
            img_list_boxes.append([x.item() for x in box_3d[img_idx, box_idx, :]])
        converted_list_boxes.append(img_list_boxes)

    return converted_list_boxes

def save_checkpoint(state, fname="my_checkpoint.pth.tar"):
    print("Saving checkpoint...")
    torch.save(state, fname)

def load_checkpoint(checkpoint, model, optimizer):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == '__main__':
    pass
