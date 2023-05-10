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

sample_random = random.Random()
# sample_random.seed(123456)


class SiamFCDataset(Dataset):
    def __init__(self, cfg, logger):
        super(SiamFCDataset, self).__init__()
        # pair information
        self.logger = logger
        self.template_size = cfg.SIAMFC.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.SIAMFC.TRAIN.SEARCH_SIZE
        self.size = 5#(self.search_size - self.template_size) // cfg.SIAMFC.TRAIN.STRIDE + 1   # from cross-correlation#5
        self.color = cfg.SIAMFC.DATASET.COLOR
        self.flip = cfg.SIAMFC.DATASET.FLIP
        self.rotation = cfg.SIAMFC.DATASET.ROTATION
        self.blur = cfg.SIAMFC.DATASET.BLUR
        self.shift = cfg.SIAMFC.DATASET.SHIFT
        self.scale = cfg.SIAMFC.DATASET.SCALE

        self.transform_extra = transforms.Compose(
            [transforms.ToPILImage(), ] +
            ([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), ] if self.color > random.random() else [])
            + ([transforms.RandomHorizontalFlip(), ] if self.flip > random.random() else [])
        )

        # train data information
        self.anno = cfg.SIAMFC.DATASET.CUSTOM_TRAIN.ANNOTATION
        self.num_use = cfg.SIAMFC.TRAIN.PAIRS
        self.root = cfg.SIAMFC.DATASET.CUSTOM_TRAIN.PATH

        self.labels = json.load(open(self.anno, 'r'))
        self.videos = list(self.labels.keys())
        self.vid_num = len(self.videos)   # video number
        self.all_pairs = []
        for vid_name in self.videos:
            vid_images = self.labels[vid_name]['image_files']
            gt_rects = self.labels[vid_name]['gt_rect']
            assert len(vid_images)==len(gt_rects)
            inds = np.arange(len(gt_rects))
            pairs_inds = [(i,i+1) for i in inds[:-1]]
            for pairs in pairs_inds:
                template_im_path = os.path.join(self.root, vid_name, vid_images[pairs[0]])
                template_bbox = gt_rects[pairs[0]]
                search_im_path = os.path.join(self.root, vid_name, vid_images[pairs[1]])
                search_bbox = gt_rects[pairs[1]]
                self.all_pairs.append((template_im_path, template_bbox, search_im_path, search_bbox))
        random.shuffle(self.all_pairs)
        self.logger.info(f"LEN DATASET {len(self.all_pairs)}")

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, index):
        """
        pick a vodeo/frame --> pairs --> data aug --> label
        """
        template_im_path, template_bbox, search_im_path, search_bbox = self.all_pairs[index]

        template_image = cv2.imread(template_im_path)
        template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
        
        search_image = cv2.imread(search_im_path)
        search_image = cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB)

        template, template_bbox, _ = self._augmentation(template_image, template_bbox, self.template_size)
        search, search_bbox, dag_param = self._augmentation(search_image, search_bbox, self.search_size)
        out_label = self.create_image_grid((self.template_size, self.template_size), search_bbox, self.size)
        
        template_bbox = self._toBBox(template_image, template_bbox)
        search_bbox = self._toBBox(search_image, search_bbox)

        # from PIL image to numpy
        template = np.array(template)
        search = np.array(search)
        #out_label = self._dynamic_label([self.size, self.size], dag_param.shift)

        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])

        return template, search, out_label, np.array(search_bbox, np.float32)

    def _posNegRandom(self):
        """
        random number from [-1, 1]
        """
        return random.random() * 2 - 1.0

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

    def _crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """
        crop image
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    def _draw(self, image, box, name):
        """
        draw image for debugging
        """
        draw_image = image.copy()
        x1, y1, x2, y2 = map(lambda x:int(round(x)), box)
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0,255,0))
        cv2.circle(draw_image, (int(round(x1 + x2)/2), int(round(y1 + y2) /2)), 3, (0, 0, 255))
        cv2.putText(draw_image, '[x: {}, y: {}]'.format(int(round(x1 + x2)/2), int(round(y1 + y2) /2)), (int(round(x1 + x2)/2) - 3, int(round(y1 + y2) /2) -3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.imwrite(name, draw_image)

    # ------------------------------------
    # function for data augmentation
    # ------------------------------------
    def _augmentation(self, image, bbox, size):
        """
        data augmentation for input pairs
        """
        shape = image.shape
        param = edict()

        param.shift = (self._posNegRandom() * self.shift, self._posNegRandom() * self.shift)   # shift
        param.scale = ((1.0 + self._posNegRandom() * self.scale), (1.0 + self._posNegRandom() * self.scale))  # scale change
        img_size = np.array(image.shape[:2])
        image = cv2.resize(image, (size, size))
        new_img_size = np.array(image.shape[:2])
        ratio = new_img_size / img_size
        bbox[0]*=ratio[1]
        bbox[2]*=ratio[1]
        bbox[1]*=ratio[0]
        bbox[3]*=ratio[0]

        if self.blur > random.random():
            image = gaussian_filter(image, sigma=(1, 1, 0))

        image = self.transform_extra(image) # other data augmentation
        return image, bbox, param

    # ------------------------------------
    # function for creating training label
    # ------------------------------------ 
    def _dynamic_label(self, fixedLabelSize, c_shift, rPos=2, rNeg=0):
        if isinstance(fixedLabelSize, int):
            fixedLabelSize = [fixedLabelSize, fixedLabelSize]

        assert (fixedLabelSize[0] % 2 == 1)

        d_label = self._create_dynamic_logisticloss_label(fixedLabelSize, c_shift, rPos, rNeg)

        return d_label

    def _create_dynamic_logisticloss_label(self, label_size, c_shift, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        # the real shift is -param['shifts']
        sz_x = sz // 2 + round(-c_shift[0]) // 8  # 8 is strides
        sz_y = sz // 2 + round(-c_shift[1]) // 8

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        dist_to_center = np.abs(x) + np.abs(y)  # Block metric
        label = np.where(dist_to_center <= rPos,
                         np.ones_like(y),
                         np.where(dist_to_center < rNeg,
                                  0.5 * np.ones_like(y),
                                  np.zeros_like(y)))
        return label
    
    def create_image_grid(self, image_size, bbox, n):
        image_height, image_width = image_size[0], image_size[1]
        x_partition_size = image_width / n
        y_partition_size = image_height / n

        # get the x and y coordinates of the bbox in the image
        bbox_x1, bbox_y1, bbox_w, bbox_h = bbox
        bbox_x2, bbox_y2 = bbox_x1 + bbox_w/n, bbox_y1 + bbox_h/n

        # create a numpy array to store the results
        output_grid = np.zeros((n, n))

        # create a meshgrid of x and y coordinates
        x, y = np.meshgrid(np.arange(n), np.arange(n))
        x1 = x * x_partition_size
        y1 = y * y_partition_size
        x2 = x1 + x_partition_size
        y2 = y1 + y_partition_size

        # check if the bbox overlaps with each partition
        mask = (bbox_x1 <= x2) & (bbox_x2 >= x1) & (bbox_y1 <= y2) & (bbox_y2 >= y1)
        # set the corresponding entries in the output grid to 1
        output_grid[mask] = 1
        return output_grid


