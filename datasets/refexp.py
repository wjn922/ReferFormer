# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms_image as T


class ModulatedDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(ModulatedDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            img, target = super(ModulatedDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            coco_img = self.coco.loadImgs(image_id)[0]
            caption = coco_img["caption"]
            dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
            target = {"image_id": image_id, "annotations": target, "caption": caption}
            img, target = self.prepare(img, target)
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            target["dataset_name"] = dataset_name
            for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id"]:
                if extra_key in coco_img:
                    target[extra_key] = coco_img[extra_key] # box xyxy -> cxcywh
            # FIXME: handle "valid", since some box may be removed due to random crop
            target["valid"] = torch.tensor([1]) if len(target["area"]) != 0 else torch.tensor([0])

            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                import random
                idx = random.randint(0, self.__len__() - 1)
        return img.unsqueeze(0), target
        # return img: [1, 3, H, W], the first dimension means T = 1.


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2] # xminyminwh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # keep the valid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if caption is not None:
            target["caption"] = caption
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["valid"] = torch.tensor([1])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set, cautious):

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
    final_scales = [296, 328, 360, 392, 416, 448, 480, 512] 

    max_size = 800
    if image_set == "train":
        horizontal = [] if cautious else [T.RandomHorizontalFlip()]
        return T.Compose(
            horizontal
            + [
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600, respect_boxes=cautious),
                            T.RandomResize(final_scales, max_size=640),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([360], max_size=640),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(dataset_file, image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    dataset = dataset_file
    PATHS = {
        "train": (root / "train2014", root / dataset / f"{mode}_{dataset}_train.json"),
        "val": (root / "train2014", root / dataset / f"{mode}_{dataset}_val.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = ModulatedDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, False),
        return_masks=args.masks,
    )
    return dataset