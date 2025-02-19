#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import numpy as np

"""
计算模型的FLOPS
python get_flops.py
"""

from sold2.experiment import load_config
from sold2.model.line_matcher import LineMatcher

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')

parser = argparse.ArgumentParser(description='Train a detector')
args = parser.parse_args()
args.shape = (1, 128, 128)
args.size_divisor = 32

cfg_path = 'sold2/config/export_line_features_mini.yaml'
config = load_config(cfg_path)
# Initialize the line matcher
line_matcher = LineMatcher(
    config["model_cfg"], None, 'cpu', config["line_detector_cfg"],
    config["line_matcher_cfg"], False)
print("\t Successfully initialized model")


def main():
    c, h, w = tuple(args.shape)
    ori_shape = (c, h, w)
    divisor = args.size_divisor
    if divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor

    input_shape = (c, h, w)

    model = line_matcher.model
    model.eval()

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30

    if divisor > 0 and \
            input_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {input_shape}\n')
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
