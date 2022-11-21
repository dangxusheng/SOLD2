#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: Jory.d
@contact: 707564875@qq.com
@site: 
@software: PyCharm
@file: vis_exported_ds.py
@time: 2022/11/11 下午1:38
@desc: 可视化导出的数据集
"""

"""
python vis_exported_ds.py
"""

import os,os.path as osp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from sold2.dataset.wireframe_dataset import WireframeDataset
from sold2.dataset.holicity_dataset import HolicityDataset
from sold2.dataset.custom_tray_dataset import TrayDataset
from sold2.dataset.merge_dataset import MergeDataset
from sold2.misc.visualize_util import plot_junctions, plot_line_segments
from sold2.misc.visualize_util import plot_images, plot_keypoints

# Initialize the wireframe dataset
with open("./sold2/config/tray_dataset_train.yaml", "r") as f:
    config = yaml.safe_load(f)

# wireframe_dataset = WireframeDataset(mode="test", config=config)
wireframe_dataset = TrayDataset(mode="train", config=config)

total = len(wireframe_dataset)
print('total: ', total)

_dir = '/home/sunnypc/dangxs/datasets/Tray_Dataset/20220816_crop_img'
save_root = f'{_dir}/../20220816_exportds_sold2_show'
os.makedirs(save_root, exist_ok=True)

for i in range(total):

    if np.random.rand()>0.5: continue

    # Read in one datapoint
    # index = 4
    # index = np.random.randint(0, total)
    data1 = wireframe_dataset[i]

    print(data1.keys())

    ref_img = data1['image'].numpy().squeeze()
    ref_junc = data1['junctions'].numpy()
    ref_line_map = data1['line_map'].numpy()
    # ref_line_points = data1['line_points'].numpy()


    # # Reference data
    # ref_img = data1['ref_image'].numpy().squeeze()
    # ref_junc = data1['ref_junctions'].numpy()
    # ref_line_map = data1['ref_line_map'].numpy()
    # ref_line_points = data1['ref_line_points'].numpy()

    # # Target data
    # target_img = data1['target_image'].numpy().squeeze()
    # target_junc = data1['target_junctions'].numpy()
    # target_line_map = data1['target_line_map'].numpy()
    # target_line_points = data1['target_line_points'].numpy()

    # Draw the points and lines
    ref_img_with_junc = plot_junctions(ref_img, ref_junc, junc_size=2)
    ref_line_segments = plot_line_segments(ref_img, ref_junc, ref_line_map, junc_size=1)
    # target_img_with_junc = plot_junctions(target_img, target_junc, junc_size=2)
    # target_line_segments = plot_line_segments(target_img, target_junc, target_line_map, junc_size=1)

    # Plot the images
    plot_images([ref_img_with_junc, ref_line_segments], ['Junctions', 'Line segments'])
    # plot_images([target_img_with_junc, target_line_segments], ['Warped junctions', 'Warped line segments'])

    # Draw the line points for training
    # ref_img_with_line_points = plot_junctions(ref_img, ref_line_points, junc_size=1)
    # target_img_with_line_points = plot_junctions(target_img, target_line_points, junc_size=1)

    # Plot the images
    # plot_images([ref_img_with_line_points, target_img_with_line_points], ['Ref', 'Target'])

    # plt.show()
    plt.savefig(f"{save_root}/{i}.jpg")

