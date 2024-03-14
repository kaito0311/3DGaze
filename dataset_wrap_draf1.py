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

class BaseDataset(Dataset):

    @staticmethod
    def load_eyes3d(fname_eyes3d):
        return load_eyes3d(fname_eyes3d)

    def load_data_file(self, data_file, debug=False):
        with open(data_file, 'rb') as fi:
            data_list = cPickle.load(fi)
        if debug:
            data_list = data_list[: 3 * self.dataset_cfg.BATCH_SIZE]
        return data_list

    def load_single_img_info(self, db_rec, element_str=None):
        if element_str:
            db_rec = db_rec[element_str]
        image_path = db_rec['image_path']
        cv_img_numpy = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        assert cv_img_numpy is not None, f"\n\nFile: {image_path} is None/Corrupted:\n"
        image_shape = np.array(cv_img_numpy.shape[0:2])
        return image_path, cv_img_numpy, image_shape

    def load_db(self, stream_db, debug= True):
        db = {}
        db_names = []
        dataset_name = "Streaming"
        # load dataset file
        data_list = [stream_db]
        # load dataset elements
        print(f"Loading streaming")
        for ii, data_sample in enumerate(tqdm.tqdm(data_list)):
            name_key, subject = self.get_name_subject(dataset_name, data_sample)
            db_names += [name_key]
            db[name_key] = dict.fromkeys(self.face_elements, {})
            for element_str in self.face_elements:
                db[name_key][element_str] = self._prepare_data(data_sample, element_str)
        return db, db_names

    def get_name_subject(self, dataset_name, data_sample):
        name_key = dataset_name + '/' + data_sample['name']
        subject = data_sample['name'].split('/')[0]
        return name_key, subject

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
            'image_path': image_path,
            'center_x': center_x,
            'center_y': center_y,
            'height': height}
        return annotation

    def _load_eye_verts(self, data_sample, element_str):
        verts = None
        if data_sample['eyes'] is not None:
            verts = (data_sample['eyes'][element_str]['P'] @ self.eye_template_homo[element_str].T).T
            verts = verts[:, [1, 0, 2]].astype(np.float32)
            verts[:, 2] -= verts[:, 2][:32].mean(axis=0)
        return verts

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

    def _load_face_gaze(self, data_sample):
        return data_sample['gaze']['face'] if data_sample['gaze'] is not None else 0

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

    def __len__(self, ):
        return len(self.db_names)



class GenericDatasetWrap(BaseDataset):
    def __init__(self, dataset_cfg=None, input_shape=None, is_train=None, transform=None,
                 fname_eyes3d='data/eyes3d.pkl', debug=False, test_sbjs=None,
                 len_datasets=1, custom_set=None, do_synthetic_training=None):

        assert dataset_cfg is not None
        self.dataset_cfg = dataset_cfg
        self.len_datasets = len_datasets
        self.is_train = is_train
        self.do_synthetic_training = do_synthetic_training

        self.test_sbjs = test_sbjs
        self.transform = transform

        # TODO make it more elegant later, individualize the augmentations
        self.aug_config = self.dataset_cfg.AUGMENTATION if self.is_train else None
        self.crop_width, self.crop_height = input_shape[0], input_shape[1]

        self.mean = torch.as_tensor([0.485, 0.456, 0.406])
        self.std = torch.as_tensor([0.229, 0.224, 0.225])
        self.aspect_ratio = self.crop_width * 1.0 / self.crop_height


        self.img_prefix = None

        self.gt_bbox_exists = True  # flag value could change
        self.get_bbox_eyes_center_func = get_pred_center
        print(self.get_bbox_eyes_center_func)
        eyes3d_dict = self.load_eyes3d(fname_eyes3d)
        self.iris_idxs481 = eyes3d_dict['iris_idxs481']
        self.trilist_eye = eyes3d_dict['trilist_eye']
        self.eye_template_homo = eyes3d_dict['eye_template_homo']

        self.idx_nosetip_in_lms68 = 30
        self.face_elements = ['left_eye', 'right_eye', 'face']

        # self.db, self.db_names = self.load_db(debug)

        # self.dataset_len = len(self.db_names)
        # self.unique_len = len(list(self.db.keys()))

    def __getitem__(self, idx):
        if not self.is_train:
            return self.get_test_data(idx)

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

    def get_test_data(self, stream_record_raw):
        db, db_names = self.load_db(stream_db= stream_record_raw, debug= False)
        db_rec = copy.deepcopy(db[db_names[0]])
        meta = []
        input_list = []
        single_image_info = self.load_single_img_info(db_rec['face'])
        image_path, cv_img, image_shape = single_image_info

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
                'image_path': image_path,
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

    