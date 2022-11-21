#!/bin/bash


# Step 4: Train the detector on the Wireframe dataset
python -m sold2.experiment --mode train --dataset_config sold2/config/wireframe_dataset_no_gt_mini_train.yaml \
--model_config sold2/config/train_detector_mini.yaml --exp_name sold2_wireframe \
--pretrained --pretrained_path experiments/sold2_synth_superpoint --checkpoint_name checkpoint-epoch105-end.tar

python -m sold2.experiment --mode train --dataset_config sold2/config/yorkUrban_dataset_no_gt_mini_train.yaml \
--model_config sold2/config/train_detector_mini.yaml --exp_name sold2_yorkUrban \
--pretrained --pretrained_path experiments/sold2_synth_superpoint --checkpoint_name checkpoint-epoch105-end.tar

python -m sold2.experiment --mode train --dataset_config sold2/config/tray_dataset_train.yaml \
--model_config sold2/config/train_detector_mini.yaml --exp_name sold2_tray \
--pretrained --pretrained_path experiments/sold2_synth_superpoint --checkpoint_name checkpoint-epoch105-end.tar


python -m sold2.experiment --mode train --dataset_config sold2/config/merge_dataset_mini.yaml \
--model_config sold2/config/train_detector_mini.yaml --exp_name sold2_synth_superpoint_128x128_ft2_merge_dataset \
--pretrained --pretrained_path experiments/sold2_synth_superpoint_128x128_ft1 --checkpoint_name checkpoint-epoch110-end.tar
