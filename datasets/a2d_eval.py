"""
This file contains implementations for the precision@k and IoU (mean, overall) evaluation metrics.
copy-paste from https://github.com/mttr2021/MTTR/blob/main/metrics.py
"""
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.mask import decode
import numpy as np

from torchvision.ops.boxes import box_area

def compute_bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    # both boxes: xyxy
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = (inter+1e-6) / (union+1e-6)
    return iou, inter, union

def compute_mask_iou(outputs: torch.Tensor, labels: torch.Tensor, EPS=1e-6):
    outputs = outputs.int()
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0
    iou = (intersection + EPS) / (union + EPS)  # EPS is used to avoid division by zero
    return iou, intersection, union

# mask
def calculate_precision_at_k_and_iou_metrics(coco_gt: COCO, coco_pred: COCO):
    print('evaluating mask precision@k & iou metrics...')
    counters_by_iou = {iou: 0 for iou in [0.5, 0.6, 0.7, 0.8, 0.9]}
    total_intersection_area = 0
    total_union_area = 0
    ious_list = []
    for instance in tqdm(coco_gt.imgs.keys()):  # each image_id contains exactly one instance
        gt_annot = coco_gt.imgToAnns[instance][0]
        gt_mask = decode(gt_annot['segmentation'])
        pred_annots = coco_pred.imgToAnns[instance]
        pred_annot = sorted(pred_annots, key=lambda a: a['score'])[-1]  # choose pred with highest score
        pred_mask = decode(pred_annot['segmentation'])
        iou, intersection, union = compute_mask_iou(torch.tensor(pred_mask).unsqueeze(0),
                                               torch.tensor(gt_mask).unsqueeze(0))
        iou, intersection, union = iou.item(), intersection.item(), union.item()
        for iou_threshold in counters_by_iou.keys():
            if iou > iou_threshold:
                counters_by_iou[iou_threshold] += 1
        total_intersection_area += intersection
        total_union_area += union
        ious_list.append(iou)
    num_samples = len(ious_list)
    precision_at_k = np.array(list(counters_by_iou.values())) / num_samples
    overall_iou = total_intersection_area / total_union_area
    mean_iou = np.mean(ious_list)
    return precision_at_k, overall_iou, mean_iou

# bbox
def calculate_bbox_precision_at_k_and_iou_metrics(coco_gt: COCO, coco_pred: COCO):
    print('evaluating bbox precision@k & iou metrics...')
    counters_by_iou = {iou: 0 for iou in [0.5, 0.6, 0.7, 0.8, 0.9]}
    total_intersection_area = 0
    total_union_area = 0
    ious_list = []
    for instance in tqdm(coco_gt.imgs.keys()):  # each image_id contains exactly one instance
        gt_annot = coco_gt.imgToAnns[instance][0]
        gt_bbox = gt_annot['bbox'] # xywh
        gt_bbox = [
            gt_bbox[0],
            gt_bbox[1],
            gt_bbox[2] + gt_bbox[0],
            gt_bbox[3] + gt_bbox[1],
        ]
        pred_annots = coco_pred.imgToAnns[instance]
        pred_annot = sorted(pred_annots, key=lambda a: a['score'])[-1]  # choose pred with highest score
        pred_bbox = pred_annot['bbox']  # xyxy
        iou, intersection, union = compute_bbox_iou(torch.tensor(pred_bbox).unsqueeze(0),
                                               torch.tensor(gt_bbox).unsqueeze(0))
        iou, intersection, union = iou.item(), intersection.item(), union.item()
        for iou_threshold in counters_by_iou.keys():
            if iou > iou_threshold:
                counters_by_iou[iou_threshold] += 1
        total_intersection_area += intersection
        total_union_area += union
        ious_list.append(iou)
    num_samples = len(ious_list)
    precision_at_k = np.array(list(counters_by_iou.values())) / num_samples
    overall_iou = total_intersection_area / total_union_area
    mean_iou = np.mean(ious_list)
    return precision_at_k, overall_iou, mean_iou
