import numpy as np
import _init_paths
import os
import cv2
import json
import random
import argparse
import numpy as np

import models.models as models
from os.path import exists, join
from torch.autograd import Variable
from tracker.siamfc import SiamFC
from tracker.siamrpn import SiamRPN
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from dataset.test_dataset import SiamFCTestDataset
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou
from core.eval_otb import overlap_ratio


def compute_overlap_success(gt_bb, result_bb, metrics: dict):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    iou_list = overlap_ratio(gt_bb, result_bb).diagonal()
    assert len(iou_list) == len(gt_bb)
    for i in range(len(iou_list)):
        success_overlap = len([x for x in thresholds_overlap if x < iou_list[i]])
        error_overlap = len(thresholds_overlap) - success_overlap
        metrics['tp']+=len(success_overlap)
        metrics['fp']+=len(error_overlap)
        
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
    init_frame_ind = -1
    target_bboxes, predict_bboxes = [], []
    metrics = {'tp':0, 'fp':0}
    for data in vid_loader:
        pair_inds, template, search = data
        template_pos, template_sz, template_image = template
        template_pos, template_sz, template_image = template_pos.numpy(), template_sz.numpy(), template_image.numpy()
        target_bbox, search_image = search
        search_image = search_image.numpy()
        if pair_inds[0]!=init_frame_ind:
            state_template = tracker.init(
                template_image[0], template_pos[0], template_sz[0], model
            )
            if len(target_bboxes)!=0 and len(predict_bboxes)!=0:
                assert len(target_bboxes) == len(predict_bboxes)
                metrics = compute_overlap_success(np.array(target_bbox), np.array(predict_bboxes), metrics)
            init_frame_ind = pair_inds[0]
            target_bboxes, predict_bboxes = [], []
        target_bboxes.append(target_bbox)
        state_pred = tracker.track(state_template, search_image[0])
        pred_location = cxy_wh_2_rect(state_pred['target_pos'], state_pred['target_sz'])    
        predict_bboxes.append(pred_location)
    
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
    for vid_name in vids_names:
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
        for k in list(mean_precision.keys()):
            mean_precision[k]+=vid_metrics[k]
        per_vid_precision.update({vid_name:vid_precision})
    
    mean_precision_value = mean_precision['tp'] / (mean_precision['tp'] + mean_precision['fp'] + 1e-9)
    logger.info(f"mean_precision_value = {mean_precision_value}")
    logger.info(f"per_vid_precision = {per_vid_precision}")
    return mean_precision_value
        
    
    
        



 