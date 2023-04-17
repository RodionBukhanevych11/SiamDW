# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang and Houwen Peng
# Email: houwen.peng@microsoft.com
# Details: siamfc dataset generator
# ------------------------------------------------------------------------------


from __future__ import division

import cv2
import json
import torch
import random
import logging
import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
from os.path import join
from easydict import EasyDict as edict
from torch.utils.data import Dataset

import sys
sys.path.append('../')
from utils.utils import *
from core.config import config


class SiamFCTestDataset(Dataset):
    def __init__(self, cfg, root_path, logger, annot_path, vid_name):
        super(SiamFCTestDataset, self).__init__()
        # pair information
        self.logger = logger
        self.template_size = cfg.SIAMFC.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.SIAMFC.TRAIN.SEARCH_SIZE
        self.size = 5#(self.search_size - self.template_size) // cfg.SIAMFC.TRAIN.STRIDE + 1   # from cross-correlation

        # aug information
        self.scale = cfg.SIAMFC.DATASET.SCALE

        self.transform_extra = transforms.Compose(
            [transforms.ToPILImage()]
        )
        
        self.anno = annot_path
        self.vid_name = vid_name
        self.root = os.path.join(root_path, vid_name)

        self.annot_json = json.load(open(self.anno, 'r'))
        self.vid_annot = self.annot_json[vid_name]
        self.image_files = self.vid_annot['image_files']
        self.gt_rects = self.vid_annot['gt_rect']
        assert len(self.image_files)==len(self.gt_rects)
        self.indexs_combination = get_index_combinations(len(self.image_files))

    def __len__(self):
        return len(self.indexs_combination)
    
    def _get_pos_sz(self, bbox):
        lx, ly = bbox[0], bbox[1]
        w, h = bbox[2] - lx, bbox[3] - ly
        pos = np.array([lx + w / 2, ly + h / 2])
        sz = np.array([w, h])
        return pos, sz
    
    def _toBBox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_size
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __getitem__(self, index):
        pair_inds = self.indexs_combination[index]
        template = (
            os.path.join(self.root, self.image_files[pair_inds[0]]),
            self.gt_rects[pair_inds[0]]
        )
        search = (
            os.path.join(self.root, self.image_files[pair_inds[1]]),
            self.gt_rects[pair_inds[1]]
        )

        template_image = cv2.imread(template[0])
        template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
        
        search_image = cv2.imread(search[0])
        search_image = cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB)
        
        template_box = template[1]
        template_pos = [(template_box[2]+template_box[0])/2, (template_box[3]+template_box[1])/2]
        template_sz = [template_box[2] - template_box[0], template_box[3]- template_box[1]]
        search[1][2], search[1][3] = search[1][2] - search[1][0], search[1][3] - search[1][1]
        template_pos, template_sz = self._get_pos_sz(template_box)

        return pair_inds, (template_pos, template_sz, template_image), \
                (search[1], search_image)
    