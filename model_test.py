#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: Jory.d
@contact: 707564875@qq.com
@site: 
@software: PyCharm
@file: model_test.py
@time: 2022/11/4 下午1:26
@desc:  模型测试
"""

import sys
sys.path.append('./')

import os, os.path as osp
import cv2
import glob
import numpy as np
import torch
import copy

from sold2.experiment import load_config
from sold2.model.line_matcher import LineMatcher

"""
python3 model_test.py
"""


def test_line_detect():
    # ckpt_path = './pretrained_models/sold2_wireframe.tar'
    # cfg_path = 'sold2/config/export_line_features2.yaml'
    # ckpt_path = './experiments/sold2_synth_superpoint_ft1/checkpoint-epoch135-end.tar'
    # ckpt_path = './experiments/sold2_synth_superpoint_128x128_ft1/checkpoint-epoch110-end.tar'
    # cfg_path = 'sold2/config/export_line_features_mini.yaml'
    #
    #
    # ckpt_path = './experiments/sold2_synth_superpoint_128x128_ft2_merge_dataset/checkpoint-epoch090-end.tar'
    # cfg_path = 'sold2/config/export_line_features_mini.yaml'


    ckpt_path = './experiments/sold2_synth_superpoint_128x128_ft1_full/checkpoint-epoch085-end.tar'
    cfg_path = 'sold2/config/export_line_features_mini.yaml'

    # Get the model config, extension and checkpoint path
    config = load_config(cfg_path)
    ckpt_path = os.path.abspath(ckpt_path)
    multiscale = False

    _dir = '/home/sunnypc/dangxs/datasets/Tray_Dataset/20220816_crop_img'
    # _dir = '/home/sunnypc/dangxs/datasets/wireframe//v1.1/test'
    images_filepath_list = glob.glob(f'{_dir}/**/*.png', recursive=True)

    save_root = f'{_dir}/../20220816_crop_img_tmp2'
    os.makedirs(save_root, exist_ok=True)

    # Get the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize the line matcher
    line_matcher = LineMatcher(
        config["model_cfg"], ckpt_path, device, config["line_detector_cfg"],
        config["line_matcher_cfg"], multiscale)
    print("\t Successfully initialized model")

    images_filepath_list=[
        './1514.png'
    ]
    # Run the inference on each image and write the output on disk
    for img_path in images_filepath_list:
        print('readfile: ', img_path)
        img = cv2.imread(img_path)
        img_copy = copy.deepcopy(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # TODO:  图像保边锐化
        # img_grap_sharp = cv2.bilateralFilter(img_gray, 3, 50, 5)
        # img_gray = img_grap_sharp

        h, w = img.shape[:2]
        # scale_factor = int(h//512+0.5)
        scale_factor = 1  # we recommend resizing the images to a resolution in the range 400~800 pixels
        # img1 = cv2.resize(img_gray, (img_gray.shape[1] // scale_factor, img_gray.shape[0] // scale_factor),
        #                   interpolation=cv2.INTER_AREA)

        img1 = cv2.resize(img_gray, (128,128))
        # img1 = img_gray

        img1 = (img1 / 255.).astype(float)
        torch_img1 = torch.tensor(img1, dtype=torch.float, device=device)[None, None]

        # Run the line detection and description
        ref_detection = line_matcher.line_detection(torch_img1, desc_only=False)

        ref_line_seg = ref_detection["line_segments"]  # [N,2,2]
        ref_junctions = ref_detection['junctions']  # [N,2]
        print('ref_line_seg.shape: ', ref_line_seg.shape)
        print('ref_junctions.shape: ', ref_junctions.shape)

        # Round and convert junctions to int (and check the boundary)
        H, W = h, w
        junctions = (np.round(ref_junctions)).astype(np.int32)
        junctions[junctions < 0] = 0
        junctions[junctions[:, 0] >= H, 0] = H - 1  # (first dim) max bounded by H-1
        junctions[junctions[:, 1] >= W, 1] = W - 1  # (second dim) max bounded by W-1
        print(ref_line_seg.shape)
        print(ref_junctions.shape)

        ref_descriptors = ref_detection["descriptor"][0].cpu().numpy()  # [N,H,W]
        print(ref_descriptors.shape)

        ref_line_seg = ref_line_seg[..., ::-1]
        for line in ref_line_seg:
            line = np.int0(line).reshape(-1)
            x1, y1, x2, y2 = line
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        junctions = junctions[:, ::-1]
        for junc in junctions:
            x, y = list(map(int, junc))
            cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

        # img = np.concatenate([img_copy, img], axis=1)
        # _savepath = f'{save_root}/{img_path[len(_dir) + 1:]}'
        # os.makedirs(osp.dirname(_savepath), exist_ok=True)
        # cv2.imwrite(_savepath, img)
        # print('savefile: ', _savepath)

        cv2.imwrite('./1514-out.jpg', img)


if __name__ == "__main__":
    print('hello')
    test_line_detect()
