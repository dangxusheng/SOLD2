#!/bin/bash

# step1:  Train on a synthetic dataset
python -m sold2.experiment --mode train --dataset_config sold2/config/synthetic_dataset_mini.yaml \
--resume --resume_path experiments/sold2_synth_superpoint_128x128_ft1 --checkpoint_name checkpoint-epoch110-end.tar \
--model_config sold2/config/train_detector_mini.yaml --exp_name sold2_synth_superpoint_128x128_ft2


#python -m sold2.experiment --mode train --dataset_config sold2/config/synthetic_dataset_mini.yaml \
#--model_config sold2/config/train_detector_mini.yaml --exp_name sold2_synth_superpoint_128x128
