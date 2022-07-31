# ------------------------------------------------------------------------
# Modified from SeqFormer (https://github.com/wjf5203/SeqFormer)
# ------------------------------------------------------------------------
# Modified from STEm-Seg (https://github.com/sabarim/STEm-Seg)
# ------------------------------------------------------------------------


import imgaug
import imgaug.augmenters as iaa
import numpy as np

from datetime import datetime

from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class ImageToSeqAugmenter(object):
    def __init__(self, perspective=True, affine=True, motion_blur=True,
                 brightness_range=(-50, 50), hue_saturation_range=(-15, 15), perspective_magnitude=0.12,
                 scale_range=1.0, translate_range={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, rotation_range=(-20, 20),
                 motion_blur_kernel_sizes=(7, 9), motion_blur_prob=0.5):

        self.basic_augmenter = iaa.SomeOf((1, None), [
                iaa.Add(brightness_range),
                iaa.AddToHueAndSaturation(hue_saturation_range)
            ]
        )

        transforms = []
        if perspective:
            transforms.append(iaa.PerspectiveTransform(perspective_magnitude))
        if affine:
            transforms.append(iaa.Affine(scale=scale_range,
                                         translate_percent=translate_range,
                                         rotate=rotation_range,
                                         order=1,  # cv2.INTER_LINEAR
                                         backend='auto'))
        transforms = iaa.Sequential(transforms)
        transforms = [transforms]

        if motion_blur:
            blur = iaa.Sometimes(motion_blur_prob, iaa.OneOf(
                [
                    iaa.MotionBlur(ksize)
                    for ksize in motion_blur_kernel_sizes
                ]
            ))
            transforms.append(blur)

        self.frame_shift_augmenter = iaa.Sequential(transforms)

    @staticmethod
    def condense_masks(instance_masks):
        condensed_mask = np.zeros_like(instance_masks[0], dtype=np.int8)
        for instance_id, mask in enumerate(instance_masks, 1):
            condensed_mask = np.where(mask, instance_id, condensed_mask)

        return condensed_mask

    @staticmethod
    def expand_masks(condensed_mask, num_instances):
        return [(condensed_mask == instance_id).astype(np.uint8) for instance_id in range(1, num_instances + 1)]

    def __call__(self, image, masks=None, boxes=None):
        det_augmenter = self.frame_shift_augmenter.to_deterministic()


        if masks is not None:
            masks_np, is_binary_mask = [], []
            boxs_np = []

            for mask in masks:
                
                if isinstance(mask, np.ndarray):
                    masks_np.append(mask.astype(np.bool))
                    is_binary_mask.append(False)
                else:
                    raise ValueError("Invalid mask type: {}".format(type(mask)))

            num_instances = len(masks_np)
            masks_np = SegmentationMapsOnImage(self.condense_masks(masks_np), shape=image.shape[:2])
            # boxs_np = BoundingBoxesOnImage(boxs_np, shape=image.shape[:2])

            seed = int(datetime.now().strftime('%M%S%f')[-8:])
            imgaug.seed(seed)
            aug_image, aug_masks = det_augmenter(image=self.basic_augmenter(image=image) , segmentation_maps=masks_np)
            imgaug.seed(seed)
            invalid_pts_mask = det_augmenter(image=np.ones(image.shape[:2] + (1,), np.uint8)).squeeze(2)
            aug_masks = self.expand_masks(aug_masks.get_arr(), num_instances)
            # aug_boxes = aug_boxes.remove_out_of_image().clip_out_of_image()
            aug_masks = [mask for mask, is_bm in zip(aug_masks, is_binary_mask)]
            return aug_image, aug_masks #, aug_boxes.to_xyxy_array()

        else:
            masks = [SegmentationMapsOnImage(np.ones(image.shape[:2], np.bool), shape=image.shape[:2])]
            aug_image, invalid_pts_mask = det_augmenter(image=image, segmentation_maps=masks)
            return aug_image, invalid_pts_mask.get_arr() == 0
