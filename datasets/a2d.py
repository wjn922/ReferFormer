"""
A2D-Sentences data loader
modified from https://github.com/mttr2021/MTTR/blob/main/datasets/a2d_sentences/a2d_sentences_dataset.py
"""
from pathlib import Path

import torch
from torchvision.io import read_video
import torchvision.transforms.functional as F

from torch.utils.data import Dataset
import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random

import h5py
from pycocotools.mask import encode, area


def get_image_id(video_id, frame_idx, ref_instance_a2d_id):
    image_id = f'v_{video_id}_f_{frame_idx}_i_{ref_instance_a2d_id}'
    return image_id

class A2DSentencesDataset(Dataset):
    """
    A Torch dataset for A2D-Sentences.
    For more information check out: https://kgavrilyuk.github.io/publication/actor_action/ or the original paper at:
    https://arxiv.org/abs/1803.07485
    """
    def __init__(self, image_folder: Path, ann_file: Path, transforms, return_masks: bool, 
                 num_frames: int, max_skip: int, subset):
        super(A2DSentencesDataset, self).__init__()
        dataset_path = str(image_folder)
        self.mask_annotations_dir = os.path.join(dataset_path, 'text_annotations/a2d_annotation_with_instances')
        self.videos_dir = os.path.join(dataset_path, 'Release/clips320H')
        self.ann_file = ann_file
        self.text_annotations = self.get_text_annotations()

        self._transforms = transforms    
        self.return_masks = return_masks # not used
        self.num_frames = num_frames     
        self.max_skip = max_skip
        self.subset = subset

        print(f'\n {subset} sample num: ', len(self.text_annotations))  
        print('\n')  

    def get_text_annotations(self):
        with open(str(self.ann_file), 'r') as f:
            text_annotations_by_frame = [tuple(a) for a in json.load(f)]
            return text_annotations_by_frame

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 

    def __len__(self):
        return len(self.text_annotations)
    
    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            text_query, video_id, frame_idx, instance_id = self.text_annotations[idx]

            text_query = " ".join(text_query.lower().split())  # clean up the text query

            # read the source window frames:
            video_frames, _, _ = read_video(os.path.join(self.videos_dir, f'{video_id}.mp4'), pts_unit='sec')  # (T, H, W, C)
            vid_len = len(video_frames)
            # note that the original a2d dataset is 1 indexed, so we have to subtract 1 from frame_idx
            frame_id = frame_idx - 1

            if self.subset == 'train':
                # get a window of window_size frames with frame frame_id in the middle.
                num_frames = self.num_frames
                # random sparse sample
                sample_indx = [frame_id]
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)

                # global sampling
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >=global_n:  # sample long range global frames
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))           
                        for s_id in select_id:                                                                   
                            sample_indx.append(all_inds[s_id])
                sample_indx.sort()
                # find the valid frame index in sampled frame list, there is only one valid frame
                valid_indices = sample_indx.index(frame_id)

            elif self.subset == 'val':
                start_idx, end_idx = frame_id - self.num_frames // 2, frame_id + (self.num_frames + 1) // 2
                sample_indx = []
                for i in range(start_idx, end_idx):
                    i = min(max(i, 0), len(video_frames)-1)  # pad out of range indices with edge frames
                    sample_indx.append(i)
                sample_indx.sort()
                # find the valid frame index in sampled frame list, there is only one valid frame
                valid_indices = sample_indx.index(frame_id)


            # read frames 
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                img = F.to_pil_image(video_frames[frame_indx].permute(2, 0, 1))
                imgs.append(img)

            # read the instance mask
            frame_annot_path = os.path.join(self.mask_annotations_dir, video_id, f'{frame_idx:05d}.h5')
            f = h5py.File(frame_annot_path)
            instances = list(f['instance'])
            instance_idx = instances.index(instance_id)  # existence was already validated during init

            instance_masks = np.array(f['reMask'])
            if len(instances) == 1:
                instance_masks = instance_masks[np.newaxis, ...]
            instance_masks = torch.tensor(instance_masks).transpose(1, 2)
            mask_rles = [encode(mask) for mask in instance_masks.numpy()]
            mask_areas = area(mask_rles).astype(np.float)
            f.close()

            # select the referred mask
            label = torch.tensor(0, dtype=torch.long)
            mask = instance_masks[instance_idx].numpy()
            if (mask > 0).any():
                y1, y2, x1, x2 = self.bounding_box(mask)
                box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                valid.append(1)
            else: # some frame didn't contain the instance
                box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                valid.append(0)
            mask = torch.from_numpy(mask)
            labels.append(label)
            boxes.append(box)
            masks.append(mask)

            # transform
            h, w = instance_masks.shape[-2:]
            labels = torch.stack(labels, dim=0) 
            boxes = torch.stack(boxes, dim=0) 
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0) 
            # there is only one valid frame
            target = {
                'frames_idx': torch.tensor(sample_indx), # [T,]
                'valid_indices': torch.tensor([valid_indices]),
                'labels': labels,                        # [1,]
                'boxes': boxes,                          # [1, 4], xyxy
                'masks': masks,                          # [1, H, W]
                'valid': torch.tensor(valid),            # [1,]
                'caption': text_query,
                'orig_size': torch.as_tensor([int(h), int(w)]), 
                'size': torch.as_tensor([int(h), int(w)]),
                'image_id': get_image_id(video_id,frame_idx, instance_id)
            }

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            imgs, target = self._transforms(imgs, target) 
            imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
            
            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            normalize,
        ])
    
    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.a2d_path)
    assert root.exists(), f'provided A2D-Sentences path {root} does not exist'
    PATHS = {
        "train": (root, root / "a2d_sentences_single_frame_train_annotations.json"),
        "val": (root, root / "a2d_sentences_single_frame_test_annotations.json"),   
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = A2DSentencesDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size), 
                                return_masks=args.masks, num_frames=args.num_frames, max_skip=args.max_skip, subset=image_set)
    return dataset