import numpy as np
import _init_paths
import os, time
import cv2
import json
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
import models.models as models
from os.path import exists, join
from torch.autograd import Variable
from tracker.siamfc import SiamFC
from tracker.siamrpn import SiamRPN
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from dataset.test_dataset import SiamFCTestDataset
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou, iou_batch


def bbox_iou_numpy(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def iou_lists(gt_bbs, result_bbs):
    iou_list = []
    for i in range(len(gt_bbs)):
        iou = bbox_iou_numpy(gt_bbs[i], result_bbs[i])
        iou_list.append(iou)
    return iou_list


def compute_overlap_success(gt_bbs, result_bbs, metrics: dict, logger):
    for i in range(len(gt_bbs)):
        gt_bb = gt_bbs[i]
        res_bb = result_bbs[i]
        res_center = (res_bb[0]+res_bb[2])//2, (res_bb[1]+res_bb[3])//2
        cntr_in_bbox = gt_bb[0]<=res_center[0]<=gt_bb[1] and gt_bb[1]<=res_center[1]<=gt_bb[3]
        if cntr_in_bbox:
            metrics['tp'] += 1
        else:
            metrics['fp'] += 1
        
    return metrics


def eval_vid(
    config, 
    tracker,
    model,
    logger, 
    vid_name,
    root_path,
    annot_path
):
    vid_set = SiamFCTestDataset(config, root_path, logger, annot_path, vid_name)
    vid_loader = DataLoader(vid_set, batch_size=1, num_workers=config.WORKERS,
                                  pin_memory=True, sampler=None)
    metrics = {'tp':0, 'fp':0}
    for data in vid_loader:
        template_pos, template_sz, template_image, search_images, search_bboxes = data
        template_pos, template_sz = template_pos[0].numpy(), template_sz[0].numpy()
        template_image = template_image[0].numpy()
        search_images, target_bboxes = search_images[0].numpy(), search_bboxes[0].numpy()
        predict_bboxes = []
        state_pred = tracker.init(
                template_image, template_pos, template_sz, model
            )
        for search_im in search_images:
            state_pred = tracker.track(state_pred, search_im)
            pred_location = cxy_wh_2_rect(state_pred['target_pos'], state_pred['target_sz'])   
            pred_location[2], pred_location[3] = pred_location[2] + pred_location[0], pred_location[3] + pred_location[1] 
            predict_bboxes.append(pred_location)
        
        target_bboxes = np.array(target_bboxes).astype(int)
        predict_bboxes = np.array(predict_bboxes).astype(int)
        assert target_bboxes.shape == predict_bboxes.shape
        metrics = compute_overlap_success(target_bboxes, predict_bboxes, metrics, logger)
        
    return metrics


def eval_dataset(
    config,
    annot_path,
    root_path,
    tracker,
    model,
    logger
):
    model.eval()
    labels = json.load(open(annot_path, 'r'))
    vids_names = list(labels.keys())
    per_vid_precision = {}
    mean_precision = {'tp':0,'fp':0}
    for vid_name in tqdm(vids_names[:]):
        vid_metrics = eval_vid(
            config, 
            tracker,
            model,
            logger, 
            vid_name,
            root_path,
            annot_path
        )
        vid_precision = vid_metrics['tp'] / (vid_metrics['tp'] + vid_metrics['fp'] + 1e-9)
        vid_precision = round(vid_precision,3)
        for k in list(mean_precision.keys()):
            mean_precision[k]+=vid_metrics[k]
        per_vid_precision.update({vid_name:vid_precision})
    
    mean_precision_value = mean_precision['tp'] / (mean_precision['tp'] + mean_precision['fp'] + 1e-9)
    mean_precision_value = round(mean_precision_value,3)
    logger.info(f"mean_precision_value = {mean_precision_value}")
    logger.info(f"per_vid_precision = {per_vid_precision}")
    return mean_precision_value
        
    
    
        



 