#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: Jory.d
@contact: 707564875@qq.com
@site: 
@software: PyCharm
@file: to_onnx.py
@time: 2022/11/11 上午10:46
@desc: 转换onnx
"""

"""
python to_onnx.py
"""

import os, os.path as osp
import numpy as np
import time
import cv2

import onnx
from onnxsim import simplify

import torch
from sold2.experiment import load_config
from sold2.model.line_matcher import LineMatcher

INPUT_SHAPE = (1, 128, 128)
size_divisor = 32

# # 官方的预训练大模型
# CFG_PATH = 'sold2/config/export_line_features2.yaml'
# CKPT_PATH = './pretrained_models/sold2_wireframe.tar'
# ONNX_SAVE_PATH = './pretrained_models/sold2_wireframe.onnx'

# 自己的小模型
CFG_PATH = './sold2/config/export_line_features_mini.yaml'
CKPT_PATH = './experiments/sold2_synth_superpoint_128x128_ft1_full/checkpoint-epoch085-end.tar'

ONNX_SAVE_PATH = f'{CKPT_PATH}.onnx'


def get_model():
    ckpt_path = CKPT_PATH
    ckpt_path = os.path.abspath(ckpt_path)
    cfg_path = CFG_PATH
    config = load_config(cfg_path)
    # Initialize the line matcher
    line_matcher = LineMatcher(
        config["model_cfg"], ckpt_path, 'cpu', config["line_detector_cfg"],
        config["line_matcher_cfg"], False)
    print("\t Successfully initialized model")
    return line_matcher


@torch.no_grad()
def convert_onnx():
    model = get_model().model
    model.eval()

    c, h, w = INPUT_SHAPE
    input_shape = (c, h, w)
    dummy_input = torch.randn(1, *input_shape)
    onnx_save_path = ONNX_SAVE_PATH
    # overwrite if existed
    torch.onnx.export(model, (dummy_input,), onnx_save_path,
                      opset_version=11,
                      verbose=True,
                      keep_initializers_as_inputs=True,
                      do_constant_folding=True,
                      )

    # simplify ONNX
    onnx_model = onnx.load(onnx_save_path)
    onnx.checker.check_model(onnx_model)
    print('check onnx is done.')
    model_simp, check = simplify(onnx_model)
    print('onnx simplify is done.', check)
    # test_onnx(onnx_save_path, model, (c, h, w), (0,) * 3, (255,) * 3)


@torch.no_grad()
def test_onnx(onnx_path, pytorch_model, input_size=(3, 224, 224), mean=(0.,) * 3, std=(255.,) * 3):
    import onnx
    import onnxruntime as ort

    INPUT_C, INPUT_H, INPUT_W = input_size
    MEAN, STD = mean, std

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    input_names = [_.name for _ in sess.get_inputs()]
    output_names = [_.name for _ in sess.get_outputs()]
    print(f'input_names: ', input_names)
    print(f'output_names: ', output_names)

    # read pic
    img0 = cv2.imread("./1514.png")
    # img0 = np.random.randint(0, 255, [INPUT_H, INPUT_W, 3], dtype=np.uint8)
    h, w = img0.shape[:2]
    # resize the pic
    img1 = cv2.resize(img0, (INPUT_W, INPUT_H))
    # img1 = cv2.resize(img0, (320, 640))
    ori_image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    if 1 == INPUT_C:
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2GRAY)
        input_image = ori_image.astype(np.float32) - np.asarray(MEAN[0])
        input_image /= np.asarray(STD[0])
        input_image = input_image[np.newaxis, ...]
    else:
        # [h,w,c]
        input_image = ori_image.astype(np.float32) - np.asarray([[MEAN]])
        input_image /= np.asarray([[STD]])
        input_image = np.transpose(input_image, [2, 0, 1])

    img_c, img_h, img_w = input_image.shape
    img_data = input_image[np.newaxis, :, :, :]
    print(img_data.shape)

    # pytorch forward
    if pytorch_model:
        pytorch_model.eval()
        x = torch.from_numpy(img_data).float()
        outs = pytorch_model(x)

        print("====================pytorch output=====================")
        for o in outs:
            print(o.shape)
            print(np.around(o.cpu().numpy().reshape(-1)[:10], 4))
        print("====================pytorch output=====================")

    # onnx forward
    input_feed = {}
    for n in input_names:
        input_feed[n] = img_data.astype(np.float32)

    start_time = time.time()
    outputs = [x.name for x in sess.get_outputs()]
    run_options = ort.RunOptions()
    run_options.log_severity_level = 0
    onnx_results = sess.run(None, input_feed, run_options=run_options)
    end_time = time.time()
    print("Inference Time used: ", end_time - start_time, 's')

    print("====================onnx output=====================")
    for o_name, o_result in zip(output_names, onnx_results):
        print(o_name, o_result.shape)
        print(np.around(o_result.reshape(-1)[:10], 4))
    print("====================onnx output=====================")
    print('done.')


if __name__ == '__main__':
    convert_onnx()
    exit(100)
    test_onnx(ONNX_SAVE_PATH, None, INPUT_SHAPE)
    print('done.')
