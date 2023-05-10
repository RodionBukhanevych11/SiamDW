import numpy as np
import json
import os, time
import random
import cv2
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from tools import get_pairs, get_index_combinations, find_combs


class SiameseNetworkDataset(Dataset):
    def __init__(self, images_path, annot_path, triplets: bool, simple_combs: bool, transform=None):
        # used to prepare the labels and images path
        self.root = images_path
        self.labels = json.load(open(annot_path, ))
        self.videos = list(self.labels.keys())
        self.vid_num = len(self.videos)   # video number
        self.all_pairs = []
        self.triplets = triplets
        for vid_name in self.videos[:2]:
            vid_images = self.labels[vid_name]['image_files']
            gt_rects = self.labels[vid_name]['gt_rect']
            assert len(vid_images)==len(gt_rects)
            if simple_combs:
                inds = np.arange(len(gt_rects))
                pairs_inds = [(i,i+1) for i in inds[:-1]]
            else:
                inds = get_index_combinations(len(gt_rects), 2)
                combinations = find_combs(inds)
                pairs_inds = get_pairs(combinations)
            for pairs in pairs_inds:
                template_im_path = os.path.join(self.root, vid_name, vid_images[pairs[0]])
                template_bbox = gt_rects[pairs[0]]
                search_im_path = os.path.join(self.root, vid_name, vid_images[pairs[1]])
                search_bbox = gt_rects[pairs[1]]
                self.all_pairs.append((template_im_path, template_bbox, search_im_path, search_bbox))
                
        self.transform = transform
        
    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, index):
        template_im_path, template_bbox, search_im_path, search_bbox = self.all_pairs[index]
        template_image = cv2.imread(template_im_path)
        template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
        search_image = cv2.imread(search_im_path)
        search_image = cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB)
        search_bbox = np.array(search_bbox)
        og_search_bbox = search_bbox.copy()
        similar = np.random.choice([0,1])
        pos_search_image_crop = search_image[og_search_bbox[1]:og_search_bbox[3],og_search_bbox[0]:og_search_bbox[2]]
        directions_bboxes = np.array(create_4_bboxes(og_search_bbox))
        rand_inds = random.sample(range(len(directions_bboxes)), 1)
        search_bbox = directions_bboxes[rand_inds,:].squeeze()
        neg_search_image_crop = search_image[search_bbox[1]:search_bbox[3],search_bbox[0]:search_bbox[2]]
        finding_neg_crop_i = 0
        while pos_search_image_crop.shape[:2]!=neg_search_image_crop.shape[:2]:
            rand_inds = random.sample(range(len(directions_bboxes)), 1)
            search_bbox = directions_bboxes[rand_inds,:].squeeze()
            neg_search_image_crop = search_image[search_bbox[1]:search_bbox[3],search_bbox[0]:search_bbox[2]]
            finding_neg_crop_i+=1
            if finding_neg_crop_i==len(directions_bboxes):
                neg_search_image_crop = search_image[0:100,0:100]
                break
        
        template_image_crop = template_image[template_bbox[1]:template_bbox[3],template_bbox[0]:template_bbox[2]]
        if self.transform is not None:
            template_image = self.transform(image = template_image_crop)['image']
            pos_search_image = self.transform(image = pos_search_image_crop)['image']
            neg_search_image = self.transform(image = neg_search_image_crop)['image']
        if self.triplets:
            return template_image, pos_search_image, neg_search_image, similar
        else:
            if similar:
                return template_image, pos_search_image, similar
            else:
                return template_image, neg_search_image, similar


def get_train_val_data(config):
    train_dataset = SiameseNetworkDataset(config.images_root, config.annots_train_path, config.triplets, config.simple_combs, get_transforms(config, True))
    valid_dataset = SiameseNetworkDataset(config.images_root, config.annots_val_path, config.triplets, config.simple_combs, get_transforms(config, False))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return train_loader, valid_loader

    
def get_transforms(config, train: bool):
    if train:
        return Compose([
            A.Resize(height=config.img_size['height'], width=config.img_size['width']),
            A.HorizontalFlip(p=0.25),
            A.HueSaturationValue(always_apply=False, p=0.3, hue_shift_limit=(-5, 5), sat_shift_limit=(-5, 5), val_shift_limit=(-150, 150)),
            A.ElasticTransform(always_apply=False, p=0.3, alpha=4, sigma=100, alpha_affine=10, interpolation=1, border_mode=1),
            A.MedianBlur(always_apply=False, p=0.25, blur_limit=(11, 21)),
            A.Downscale(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return Compose([
            A.Resize(height=config.img_size['height'], width=config.img_size['width']),
            A.Normalize(),
            ToTensorV2(),
        ])
        
        
def create_4_bboxes(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    bbox1 = np.array([bbox[0] - w, bbox[1], bbox[2] - w, bbox[3]])
    bbox2 = np.array([bbox[0], bbox[1] -h, bbox[2], bbox[3] - h])
    bbox3 = np.array([bbox[0] + w, bbox[1], bbox[2] + w, bbox[3]])
    bbox4 = np.array([bbox[0], bbox[1] + h, bbox[2], bbox[3] + h])
    bbox5 = np.array([bbox[0] - w, bbox[1] - h, bbox[0], bbox[1]])
    bbox6 = np.array([bbox[0] + w, bbox[1] - h, bbox[2] + w, bbox[3] - h])
    bbox7 = np.array([bbox[2], bbox[3], bbox[2] + w, bbox[3] + h])
    bbox8 = np.array([bbox[0] - w, bbox[1] + h, bbox[2] - w, bbox[3] + h])

    # Return the four new bounding boxes
    return bbox1, bbox2, bbox3, bbox4, bbox5, bbox6, bbox7, bbox8
