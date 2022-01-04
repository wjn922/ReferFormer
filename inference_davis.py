'''
Inference code for ReferFormer, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
Ref-Davis17 does not support visualize
'''
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F
import json


import opts
from tqdm import tqdm

import multiprocessing as mp
import threading

from tools.colormap import colormap


# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

# build transform
transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    

def main(args):
    args.dataset_file = "davis"
    args.masks = True
    args.batch_size == 1
    print("Inference only supports for batch size = 1") 
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, split)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    # load data
    root = Path(args.davis_path) # data/ref-davis
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    video_list = list(data.keys())

    # create subprocess
    thread_num = args.ngpu
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = math.ceil(float(video_num) / float(thread_num))

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, data, 
                                                   save_path_prefix, save_visualize_path_prefix, 
                                                   img_folder, sub_video_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" %(total_time))


def sub_processor(lock, pid, args, data, save_path_prefix, save_visualize_path_prefix, img_folder, video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)

    # model
    model, criterion, _ = build_model(args)  
    device = args.device
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if pid == 0:
        print('number of params:', n_parameters)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
    	raise ValueError('Please specify the checkpoint for inference.')

    # get palette
    palette_img = os.path.join(args.davis_path, "valid/Annotations/blackswan/00000.png")
    palette = Image.open(palette_img).getpalette()

    # start inference
    num_all_frames = 0
    model.eval()
    
    # 1. for each video
    for video in video_list:
        metas = []

        expressions = data[video]["expressions"]   
        expression_list = list(expressions.keys()) 
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i] # start from 0
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # since there are 4 annotations
        num_obj = num_expressions // 4

        # 2. for each annotator
        for anno_id in range(4): # 4 annotators
            anno_logits = []  
            anno_masks = []   # [num_obj+1, video_len, h, w], +1 for background

            for obj_id in range(num_obj): 
                i = obj_id * 4 + anno_id
                video_name = meta[i]["video"]
                exp = meta[i]["exp"]
                exp_id = meta[i]["exp_id"]
                frames = meta[i]["frames"]

                video_len = len(frames)
                # NOTE: the im2col_step for MSDeformAttention is set as 64
                # so the max length for a clip is 64
                # store the video pred results
                all_pred_logits = []
                all_pred_masks = []

                # 3. for each clip
                for clip_id in range(0, video_len, 36):
                    frames_ids = [x for x in range(video_len)]
                    clip_frames_ids = frames_ids[clip_id : clip_id + 36]
                    clip_len = len(clip_frames_ids)

                    # load the clip images
                    imgs = []
                    for t in clip_frames_ids:
                        frame = frames[t]
                        img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                        img = Image.open(img_path).convert('RGB')
                        origin_w, origin_h = img.size
                        imgs.append(transform(img)) # list[Img]
                    
                    imgs = torch.stack(imgs, dim=0).to(args.device) # [video_len, 3, H, W]
                    img_h, img_w = imgs.shape[-2:]
                    size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
                    target = {"size": size}

                    with torch.no_grad():
                        outputs = model([imgs], [exp], [target])
                    
                    pred_logits = outputs["pred_logits"][0] # [t, q, k]
                    pred_masks = outputs["pred_masks"][0]   # [t, q, h, w]

                    # according to pred_logits, select the query index
                    pred_scores = pred_logits.sigmoid() # [t, q, k]
                    pred_scores = pred_scores.mean(0)   # [q, K]
                    max_scores, _ = pred_scores.max(-1) # [q,]
                    _, max_ind = max_scores.max(-1)     # [1,]
                    max_inds = max_ind.repeat(clip_len)
                    pred_masks = pred_masks[range(clip_len), max_inds, ...] # [t, h, w]
                    pred_masks = pred_masks.unsqueeze(0)

                    pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
                    pred_masks = pred_masks.sigmoid()[0] # [t, h, w], NOTE: here mask is score

                    # store the clip results
                    pred_logits = pred_logits[range(clip_len), max_inds] # [t, k]
                    all_pred_logits.append(pred_logits)
                    all_pred_masks.append(pred_masks)
                
                all_pred_logits = torch.cat(all_pred_logits, dim=0) # (video_len, K)
                all_pred_masks = torch.cat(all_pred_masks, dim=0)   # (video_len, h, w) 
                anno_logits.append(all_pred_logits) 
                anno_masks.append(all_pred_masks)   
            
            # handle a complete image (all objects of a annotator)
            anno_logits = torch.stack(anno_logits) # [num_obj, video_len, k]
            anno_masks = torch.stack(anno_masks)   # [num_obj, video_len, h, w]
            t, h, w = anno_masks.shape[-3:]
            anno_masks[anno_masks < 0.5] = 0.0
            background = 0.1 * torch.ones(1, t, h, w).to(args.device)
            anno_masks = torch.cat([background, anno_masks], dim=0) # [num_obj+1, video_len, h, w]
            out_masks = torch.argmax(anno_masks, dim=0) # int, the value indicate which object, [video_len, h, w]

            out_masks = out_masks.detach().cpu().numpy().astype(np.uint8) # [video_len, h, w]
    
            # save results
            anno_save_path = os.path.join(save_path_prefix, f"anno_{anno_id}", video)
            if not os.path.exists(anno_save_path):
                os.makedirs(anno_save_path)
            for f in range(out_masks.shape[0]):
                img_E = Image.fromarray(out_masks[f])
                img_E.putpalette(palette)
                img_E.save(os.path.join(anno_save_path, '{:05d}.png'.format(f)))


        with lock:
            progress.update(1)
    result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()



# Post-process functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x-10, y, x+10, y), tuple(cur_color), width=4)
        draw.line((x, y-10, x, y+10), tuple(cur_color), width=4)

def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x-2, y-2, x+2, y+2), 
                            fill=tuple(cur_color), outline=tuple(cur_color), width=1)

def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferFormer inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    main(args)

