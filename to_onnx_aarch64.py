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

INPUT_SHAPE = (1, 256, 256)
size_divisor = 32
ONNX_SAVE_PATH = './pretrained_models/sold2_wireframe.onnx'


def get_spend_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        t = (time.time() - s_time) * 1000
        print('{}() elpased：{:.3f} ms'.format(f.__name__, t))
        return res
    return inner


@get_spend_time
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
    # img0 = cv2.imread("./1.jpg")
    img0 = np.random.randint(0, 255, [INPUT_H, INPUT_W, 3], dtype=np.uint8)
    h, w = img0.shape[:2]
    # resize the pic
    img1 = cv2.resize(img0, (INPUT_W, INPUT_H))
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
    # convert_onnx()
    test_onnx(ONNX_SAVE_PATH, None, INPUT_SHAPE)
    print('done.')
