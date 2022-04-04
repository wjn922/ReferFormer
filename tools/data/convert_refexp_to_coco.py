import numpy as np
import os
from datasets.refer import REFER
import cv2  
from tqdm import tqdm
import json
import pickle
import json


def convert_to_coco(data_root='data/coco', output_root='data/coco', dataset='refcoco', dataset_split='unc'):
    dataset_dir = os.path.join(data_root, dataset)
    output_dir = os.path.join(output_root, dataset) # .json save path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read REFER
    refer = REFER(data_root, dataset, dataset_split)
    refs = refer.Refs
    anns = refer.Anns
    imgs = refer.Imgs
    cats = refer.Cats
    sents = refer.Sents
    """
    # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}

    Refs: List[Dict], "sent_ids", "file_name", "ann_id", "ref_id", "image_id", "category_id", "split", "sentences"
                      "sentences": List[Dict], "tokens"(List), "raw", "sent_id", "sent"
    Anns: List[Dict], "segmentation", "area", "iscrowd", "image_id", "bbox", "category_id", "id"
    Imgs: List[Dict], "license", "file_name", "coco_url", "height", "width", "date_captured", "flickr_url", "id"
    Cats: List[Dict], "supercategory", "name", "id"
    Sents: List[Dict], "tokens"(List), "raw", "sent_id", "sent", here the "sent_id" is consistent
    """
    print('Dataset [%s_%s] contains: ' % (dataset, dataset_split))
    ref_ids = refer.getRefIds()
    image_ids = refer.getImgIds()
    print('There are %s expressions for %s refereed objects in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

    print('\nAmong them:')
    if dataset == 'refcoco':
        splits = ['train', 'val', 'testA', 'testB']
    elif dataset == 'refcoco+':
        splits = ['train', 'val',  'testA', 'testB']
    elif dataset == 'refcocog':
        splits = ['train', 'val', 'test']  # we don't have test split for refcocog right now.

    for split in splits:
        ref_ids = refer.getRefIds(split=split)
        print('     %s referred objects are in split [%s].' % (len(ref_ids), split))

    with open(os.path.join(dataset_dir, "instances.json"), "r") as f:
        ann_json = json.load(f)


    # 1. for each split: train, val...
    for split in splits:
        max_length = 0 # max length of a sentence

        coco_ann = {
            "info": "",
            "licenses": "",
            "images": [],   # each caption is a image sample
            "annotations": [],
            "categories": []
        }
        coco_ann['info'], coco_ann['licenses'], coco_ann['categories'] = \
                                    ann_json['info'], ann_json['licenses'], ann_json['categories']
        
        num_images = 0 # each caption is a sample, create a "images" and a "annotations", since each image has one box
        ref_ids = refer.getRefIds(split=split)
        # 2. for each referred object
        for i in tqdm(ref_ids): 
            ref = refs[i]
            # "sent_ids", "file_name", "ann_id", "ref_id", "image_id", "category_id", "split", "sentences"
            #             "sentences": List[Dict], "tokens"(List), "raw", "sent_id", "sent"
            img = imgs[ref["image_id"]]
            ann = anns[ref["ann_id"]]

            # 3. for each sentence, which is a sample
            for sentence in ref["sentences"]: 
                num_images += 1
                # append image info
                image_info = {
                    "file_name": img["file_name"],
                    "height": img["height"],
                    "width": img["width"],
                    "original_id": img["id"],
                    "id": num_images,
                    "caption": sentence["sent"],
                    "dataset_name": dataset
                }
                coco_ann["images"].append(image_info)

                # append annotation info
                ann_info = {
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"],
                    "bbox": ann["bbox"],
                    "image_id": num_images,
                    "category_id": ann["category_id"],
                    "id": num_images,
                    "original_id": ann["id"]
                }
                coco_ann["annotations"].append(ann_info)

                max_length = max(max_length, len(sentence["tokens"]))
        
        print("Total expression: {} in split {}".format(num_images, split))
        print("Max sentence length of the split: ", max_length)
        # save the json file
        save_file = "instances_{}_{}.json".format(dataset, split)
        with open(os.path.join(output_dir, save_file), 'w') as f:
            json.dump(coco_ann, f)

if __name__ == '__main__':
    datasets = ["refcoco", "refcoco+", "refcocog"]
    datasets_split = ["unc", "unc", "umd"]
    for (dataset, dataset_split) in zip(datasets, datasets_split):
        convert_to_coco(dataset=dataset, dataset_split=dataset_split)
        print("")


"""
# original mapping 
{'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 
'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 
'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 
'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 
'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 
'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 
'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 
'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 
'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 
'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90}

"""
