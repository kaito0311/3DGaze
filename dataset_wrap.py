import _pickle as cPickle

import os
import cv2
import numpy as np
import tqdm
from torch.utils.data import Dataset

from lib.utils.defaults import DATASETS_WITH_SAME_TRAIN_TEST_FILES
from lib.utils import load_eyes3d
import copy
import random
import cv2
import numpy as np
import torch

from lib.utils.defaults import DATASET_INFO
from lib.utils import *


class Gaze3DDatasetWrap:
    def __init__(self, dataset_cfg=None, input_shape=None, is_train=None, transform=None,
                 fname_eyes3d='data/eyes3d.pkl', debug=False, test_sbjs=None,
                 len_datasets=1, custom_set=None, do_synthetic_training=None) -> None:
        
        assert dataset_cfg is not None
        self.dataset_cfg = dataset_cfg

        self.is_train = False 

        self.transform = transform
        
        self.crop_width, self.crop_height = input_shape[0], input_shape[1]
        self.mean = torch.as_tensor([0.485, 0.456, 0.406])
        self.std = torch.as_tensor([0.229, 0.224, 0.225])
        self.aspect_ratio = self.crop_width * 1.0 / self.crop_height

        self.gt_bbox_exists = True  # flag value could change
        self.get_bbox_eyes_center_func = get_pred_center
        eyes3d_dict = self.load_eyes3d(fname_eyes3d)
        self.iris_idxs481 = eyes3d_dict['iris_idxs481']
        self.trilist_eye = eyes3d_dict['trilist_eye']
        self.eye_template_homo = eyes3d_dict['eye_template_homo']
        
        self.idx_nosetip_in_lms68 = 30
        self.face_elements = ['left_eye', 'right_eye', 'face']
   
    @staticmethod
    def load_eyes3d(fname_eyes3d):
        return load_eyes3d(fname_eyes3d)

    def _prepare_eyes(self, data_sample, element_str):
        element_str_short = element_str.split('_')[0]
        # coords (might be None in inference datasets)
        verts = self._load_eye_verts(data_sample, element_str_short)
        # gaze (might be None in inference datasets)
        gaze = self._load_face_gaze(data_sample)
        # bbox
        center_x, center_y, width, height = self._get_bbox_center(None, None, element_str, data_sample)
        width *= self.dataset_cfg.AUGMENTATION.EXTENT_TO_CROP_RATIO
        height *= self.dataset_cfg.AUGMENTATION.EXTENT_TO_CROP_RATIO
        return center_x, center_y, width, height, verts, gaze
    
    def _load_face_verts(self, data_sample):
        if data_sample['face']['xyz68'] is not None:
            verts = data_sample['face']['xyz68']
            verts = verts[:, [1, 0, 2]].astype(np.float32)
            verts[:, 2] -= verts[:, 2][-68:][self.idx_nosetip_in_lms68]
        elif data_sample['face']['xy5'] is not None:
            verts = data_sample['face']['xy5']
            verts = verts[:, [1, 0]].astype(np.float32)
            verts = np.concatenate((verts, np.zeros((5, 1))), axis=1)
        else: 
            raise KeyError("The face sample does not contain xyz68 landmarks")
        return verts
    
    def _prepare_face(self, data_sample):
        # coords
        verts = self._load_face_verts(data_sample)
        # gaze (might be None in inference datasets)
        gaze = self._load_face_gaze(data_sample)
        # bbox
        center_x, center_y, width, height = self._get_bbox_center(verts[:, 0], verts[:, 1], 'face', data_sample)
        return center_x, center_y, width, height, verts, gaze

    def _load_face_gaze(self, data_sample):
        return data_sample['gaze']['face'] if data_sample['gaze'] is not None else 0

    def _load_eye_verts(self, data_sample, element_str):
        verts = None
        if data_sample['eyes'] is not None:
            verts = (data_sample['eyes'][element_str]['P'] @ self.eye_template_homo[element_str].T).T
            verts = verts[:, [1, 0, 2]].astype(np.float32)
            verts[:, 2] -= verts[:, 2][:32].mean(axis=0)
        return verts
    def _prepare_face(self, data_sample):
        # coords
        verts = self._load_face_verts(data_sample)
        # gaze (might be None in inference datasets)
        gaze = self._load_face_gaze(data_sample)
        # bbox
        center_x, center_y, width, height = self._get_bbox_center(verts[:, 0], verts[:, 1], 'face', data_sample)
        return center_x, center_y, width, height, verts, gaze

    def _prepare_left_eye(self, data_sample):
        return self._prepare_eyes(data_sample, 'left_eye')

    def _prepare_right_eye(self, data_sample):
        return self._prepare_eyes(data_sample, 'right_eye')


    def _prepare_data(self, data_sample: dict, element_str: str) -> dict:
        preparation_func = eval(f"self._prepare_{element_str}")
        center_x, center_y, width, height, verts, gaze = preparation_func(data_sample)
        image_path = None
        # assert os.path.exists(image_path), f"Unable to locate sample:\n{image_path}"
        head_pose = np.zeros(2)
        if 'head_pose' in data_sample['face']:
            head_pose = data_sample['face']['head_pose']
        annotation = {
            'verts': verts,
            'gaze': gaze,
            'head_pose': head_pose,
            'image_path': None,
            'center_x': center_x,
            'center_y': center_y,
            'height': height}
        return annotation

    def _get_bbox_center(self, x, y, element_str, data_sample):
        if element_str == 'face':
            return get_gt_center(x, y)
        while True:
            if not (self.dataset_cfg.USE_GT_BBOX and self.gt_bbox_exists):
                bbox_center = self.get_bbox_eyes_center_func(data_sample, element_str)
                if bbox_center is None:
                    self.gt_bbox_exists = False
                    print(f"Predicted center for bboxes of eyes does not exist for dataset {self.roots['dataset_name']}"
                          f"\nUsing gt bboxes instead..")
                    continue
                return bbox_center
            else:
                return get_gt_center(x, y)

    def __call__(self, bgr_image, stream_record_raw):
        '''
        image: BGR 
        '''
        input_list = [] 
        meta = [] 

        db_rec = dict.fromkeys(self.face_elements, {})
        for element_str in self.face_elements:
            db_rec[element_str] = self._prepare_data(stream_record_raw, element_str)
        
        cv_img = bgr_image 
        image_shape = cv_img.shape[0:2]

        for element_str in self.face_elements:
            cv_img_numpy = cv_img.copy()
            
            # load img info
            center = [db_rec[element_str]['center_x'], db_rec[element_str]['center_y']]
            height = db_rec[element_str]['height']
            width = height * self.aspect_ratio
            input_args = (cv_img_numpy, [self.crop_width, self.crop_height], 0, False)  # rot=0, Flip=False
            trans, img_patch_cv = get_input_and_transform(center, [width, height], *input_args)

            np_img_patch_copy = img_patch_cv.copy()
            np_img_patch_copy = np.transpose(np_img_patch_copy, (2, 0, 1)) / 255  # (C,H,W) and between 0,1
            img_patch_torch = torch.as_tensor(np_img_patch_copy, dtype=torch.float32)  # to torch and from int to float
            img_patch_torch.sub_(self.mean.view(-1, 1, 1)).div_(self.std.view(-1, 1, 1))
            input_list += [img_patch_torch]

            # Transform vertices
            verts = 0
            if db_rec[element_str]['verts'] is not None:
                xy = db_rec[element_str]['verts'][:, [0, 1]]
                z = db_rec[element_str]['verts'][:, 2]
                xy = affine_transform_array(xy, trans)
                z *= float(self.crop_height) / float(height)
                # set verts
                verts = np.zeros_like(db_rec[element_str]['verts'])
                verts[:, [0, 1]] = xy
                verts[:, 2] = z
            
            meta += [{
                'verts': verts, 
                'gaze': db_rec[element_str]['gaze'],
                'head_pose': db_rec[element_str]['head_pose'],
                'element': element_str,
                'image_path': None,
                'image_shape': image_shape,
                'scale_multiplier': 1.,
                'center': center,
                'width': width,
                'height': height,
                'flip': False,
                'rotation': 0,
                'trans': trans,
                'init_height': db_rec[element_str]['height']
            }]
    
        model_input = np.concatenate((input_list[0], input_list[1], input_list[2]), axis=0)

        return model_input, meta
    