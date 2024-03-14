import os
import gc
import numpy as np
import pickle
import cv2 
import torch
import torch.backends.cudnn as cudnn

from lib.utils import load_from_checkpoint, get_final_output_dir, \
                    parse_args, update_config, update_dict, update_dataset_info, config
from lib.core import inference
from lib.dataset import build_dataset
from lib.models import build_model

from lib.dataset.datasets import InferenceDataset 

from dataset_wrap import Gaze3DDatasetWrap
from lib.utils import trans_coords_from_patch_to_org_3d, estimate_affine_matrix_3d23d, load_eyes3d, get_final_results_dir, points_to_vector


def main(): 
    # setup config
    args = parse_args()
    update_config(args.cfg)
    update_dict(config, vars(args))
    update_dataset_info(args)
    device= "cpu"
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = build_model(config).to(device)    
    # load checkpoint
    final_output_dir = get_final_output_dir(config)
    checkpoint = (args.checkpoint or os.path.join(final_output_dir, 'model_best.pth'))
    print(checkpoint)
    if os.path.isfile(checkpoint):
        print(f'=> loading model from {checkpoint}')
        load_from_checkpoint(checkpoint, model, skip_optimizer=args.skip_optimizer)
    else:
        raise f'No valid checkpoints file {checkpoint}'
    
    
    dataset = InferenceDataset(
        dataset_cfg=config.DATASET,
        input_shape=[config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]],
        is_train=False,
        debug= False,
        custom_set= None,
        len_datasets=1
    )

    dataset_wrap  = Gaze3DDatasetWrap(
        dataset_cfg=config.DATASET,
        input_shape=[config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]],
        is_train=False,
        debug= False,
        custom_set= None,
        len_datasets=1
    )

    file = open("output/preprocessing/data_face68.pkl", "rb")

    data = pickle.load(file)

    bgr_image = cv2.imread("data/example_images/image37049_0.jpg")
    input_dat_wrap, meta_data_wrap = dataset_wrap(
        bgr_image, 
        data[0]
    )
    input_data, meta_data = dataset[0] 



    # ====================== 
    n_verts_eye = int(config.MODEL.NUM_POINTS_OUT_EYES / 2)

    eyes3d_dict = load_eyes3d('data/eyes3d.pkl')
    iris_idxs481 = eyes3d_dict['iris_idxs481']
    trilist_eye = eyes3d_dict['trilist_eye']
    eye_template = eyes3d_dict['eye_template']

    input_dat_wrap = torch.from_numpy(input_data).unsqueeze(0)

    with torch.no_grad():
        output = model(input_dat_wrap.to("cpu"))
    
    print(output[0].shape)
    print(output[0][0])
    
    # print(output)


if __name__ == "__main__":
    main()