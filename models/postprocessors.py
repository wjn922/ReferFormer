# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Postprocessors class to transform MDETR output according to the downstream task"""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pycocotools.mask as mask_util

from util import box_ops


class A2DSentencesPostProcess(nn.Module):
    """
    This module converts the model's output into the format expected by the coco api for the given task
    """
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, outputs, orig_target_sizes, max_target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            orig_target_sizes: original size of the samples (no augmentations or padding)
            max_target_sizes: size of samples (input to model) after size augmentation.
            NOTE: the max_padding_size is 4x out_masks.shape[-2:]
        """
        assert len(orig_target_sizes) == len(max_target_sizes)
        
        # there is only one valid frames, thus T=1
        out_logits = outputs['pred_logits'][:, 0, :, 0]  # [B, T, N, 1] -> [B, N]
        out_masks = outputs['pred_masks'][:, 0, :, :, :] # [B, T, N, out_h, out_w] -> [B, N, out_h, out_w]
        out_h, out_w = out_masks.shape[-2:]

        scores = out_logits.sigmoid()
        pred_masks = F.interpolate(out_masks, size=(out_h*4, out_w*4), mode="bilinear", align_corners=False) # [B, N, H, W]
        pred_masks = (pred_masks.sigmoid() > 0.5) # [B, N, H, W]
        processed_pred_masks, rle_masks = [], []
        # for each batch
        for f_pred_masks, resized_size, orig_size in zip(pred_masks, max_target_sizes, orig_target_sizes):
            f_mask_h, f_mask_w = resized_size  # resized shape without padding
            f_pred_masks_no_pad = f_pred_masks[:, :f_mask_h, :f_mask_w].unsqueeze(1)  # remove the samples' padding, [:, 1, h, w]
            # resize the samples back to their original dataset (target) size for evaluation
            f_pred_masks_processed = F.interpolate(f_pred_masks_no_pad.float(), size=tuple(orig_size.tolist()), mode="nearest") # origin size, [:, 1, h, w]
            f_pred_rle_masks = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                                for mask in f_pred_masks_processed.cpu()]
            processed_pred_masks.append(f_pred_masks_processed)
            rle_masks.append(f_pred_rle_masks)
        predictions = [{'scores': s, 'masks': m, 'rle_masks': rle}
                       for s, m, rle in zip(scores, processed_pred_masks, rle_masks)]
        return predictions


# PostProcess for pretraining
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        Returns:

        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        
        # coco, num_frames=1
        out_logits = outputs["pred_logits"].flatten(1, 2)
        out_boxes = outputs["pred_boxes"].flatten(1, 2)
        bs, num_queries = out_logits.shape[:2]

        prob = out_logits.sigmoid() # [bs, num_queries, num_classes]
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), k=num_queries, dim=1, sorted=True) 
        scores = topk_values # [bs, num_queries]
        topk_boxes = topk_indexes // out_logits.shape[2] # [bs, num_queries]
        labels = topk_indexes % out_logits.shape[2] # [bs, num_queries]

        boxes = box_ops.box_cxcywh_to_xyxy(out_boxes) # [bs, num_queries, 4]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :] # [bs, num_queries, 4]

        assert len(scores) == len(labels) == len(boxes)
        # binary for the pretraining
        results = [{"scores": s, "labels": torch.ones_like(l), "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results


class PostProcessSegm(nn.Module):
    """Similar to PostProcess but for segmentation masks.
    This processor is to be called sequentially after PostProcess.
    Args:
        threshold: threshold that will be applied to binarize the segmentation masks.
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        """Perform the computation
        Parameters:
            results: already pre-processed boxes (output of PostProcess) NOTE here
            outputs: raw outputs of the model
            orig_target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            max_target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                              after data augmentation.
        """
        assert len(orig_target_sizes) == len(max_target_sizes)

        out_logits = outputs["pred_logits"].flatten(1, 2)
        out_masks = outputs["pred_masks"].flatten(1, 2)
        bs, num_queries = out_logits.shape[:2]

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), k=num_queries, dim=1, sorted=True) 
        scores = topk_values # [bs, num_queries]
        topk_boxes = topk_indexes // out_logits.shape[2] # [bs, num_queries]
        labels = topk_indexes % out_logits.shape[2] # [bs, num_queries]

        outputs_masks = [out_m[topk_boxes[i]].unsqueeze(0) for i, out_m, in enumerate(out_masks)] # list[Tensor]
        outputs_masks = torch.cat(outputs_masks, dim=0) # [bs, num_queries, H, W]
        out_h, out_w = outputs_masks.shape[-2:]

        # max_h, max_w = max_target_sizes.max(0)[0].tolist() 
        # outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = F.interpolate(outputs_masks, size=(out_h*4, out_w*4), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1) # [:, 1, h, w]
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()
            results[i]["rle_masks"] = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                    for mask in results[i]["masks"].cpu()]

        return results



def build_postprocessors(args, dataset_name):
    if dataset_name == 'a2d' or dataset_name == 'jhmdb':
        postprocessors = A2DSentencesPostProcess(threshold=args.threshold)
    else:
        # for coco pretrain postprocessor
        postprocessors: Dict[str, nn.Module] = {"bbox": PostProcess()}
        if args.masks:
            postprocessors["segm"] = PostProcessSegm(threshold=args.threshold)
    return postprocessors
