import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOv1Loss(nn.Module):
    def __init__(self, s=7, b=2, c=3, lambda_coord=5, lambda_noobj=0.5, box_format="midpoint"):
        super(YOLOv1Loss, self).__init__()
        self.s = s
        self.b = b
        self.c = c
        self.box_format = box_format
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, preds, labels):
        '''
        Returns the loss of YOLOv1

        Parameters:
            preds (tensor): predicted bounding boxes in the shape
                            (batch_size, s, s, 5 * b + C)
            labels (tensor): ground truth bounding boxes in the shape
                            (batch_size, s, s, 5 * b + C)

        Returns:
            loss (float): The final loss consists of
                          1. Coordinate Loss
                          2. Confidence Loss(0bj/noobj)
                          3. Class Loss
        0:3 - class label one-hot vector
        3 - box1 confidence
        4:8 - box1 (x,y,w,h)
        8 - box2 confidence
        9:13 - box2 (x,y,h,w)
        '''

        '''
        First, we need to determine which box is responsible for detecting the
        obj in specific grid cell, given that an object exists. As stated in
        the original paper, only ONE predicted bounding box should be responsible.
        This is also a limitation of YOLOv1. The way to determine the responsibility
        is to compare both predictions' IoU with the ground truth box and pick
        the one with the highest IoU, given an object exists.
        '''
        preds = preds.reshape(-1, self.s, self.s, self.c + 5 * self.b)

        # Ground truth coordinates, class one-hot vector, confidence
        gt_coord = labels[..., 4:8]
        gt_class = labels[..., 0:3]
        gt_confidence = labels[..., 3:4]

        Iobj = labels[..., 3:4]

        # COORDINATES for box1, 2
        box1_coord = preds[..., 4:8]
        box2_coord = preds[..., 9:13]

        # CLASS LABEL one-hot vector
        pred_class = preds[..., 0:3]

        # CONFIDENCE for box 1,2
        box1_confidence = preds[..., 3:4]
        box2_confidence = preds[..., 8:9]

        # IoU score for box 1,2
        box1_iou = intersection_over_union(
            box1_coord,
            gt_coord,
            box_format=self.box_format
        )

        box2_iou = intersection_over_union(
            box2_coord,
            gt_coord,
            box_format=self.box_format
        )

        iou_combined = torch.cat(
            tensors=(box1_iou, box2_iou),
            dim = -1
        )

        # select best box with higher IoU
        # (batch_size, s, s, 1) -> 0 or 1
        best_box_num = iou_combined.argmax(
            dim = -1, keepdim=True
        )

        # BEST box confidence
        best_box_confidence = (
            (1 - best_box_num) * box1_confidence
            + best_box_num * box2_confidence
        )

        # BEST box coordinates (x,y,w,h)
        # (batch_size, s, s, 4)
        best_box_coord = (
            (1 - best_box_num) * box1_coord
            + best_box_num * box2_coord
        )

        #######################
        #   COORDINATE LOSS   #
        #######################
        torch.autograd.set_detect_anomaly(True)
        best_box_coord[..., 2:4] = torch.sign(best_box_coord[..., 2:4]) * torch.sqrt(
            torch.abs(best_box_coord[..., 2:4] + 1e-6)
        )

        gt_coord[..., 2:4] = torch.sqrt(gt_coord[..., 2:4])

        coord_loss = self.lambda_coord * self.mse(
            torch.flatten(Iobj * best_box_coord, end_dim=-2),
            torch.flatten(Iobj * gt_coord, end_dim=-2),
        )

        #######################
        #   CONFIDENCE LOSS   #
        #######################
        # If YES object
        obj_confidence_loss = self.mse(
            torch.flatten(Iobj * best_box_confidence, end_dim=-2),
            torch.flatten(Iobj * gt_confidence, end_dim=-2),
        )

        # If No object
        noobj_confidence_loss = self.mse(
            torch.flatten((1 - Iobj) * box1_confidence, end_dim=-2),
            torch.flatten((1 - Iobj) * gt_confidence, end_dim=-2),
        )

        noobj_confidence_loss += self.mse(
            torch.flatten((1 - Iobj) * box2_confidence, end_dim=-2),
            torch.flatten((1 - Iobj) * gt_confidence, end_dim=-2),
        )

        confidence_loss = (
            obj_confidence_loss + self.lambda_noobj * noobj_confidence_loss
        )

        ##################
        #   CLASS LOSS   #
        ##################

        class_loss = self.mse(
            torch.flatten(Iobj * pred_class, end_dim=-2),
            torch.flatten(Iobj * gt_class, end_dim=-2),
        )

        return coord_loss + confidence_loss + class_loss

if __name__ == '__main__':
    pass