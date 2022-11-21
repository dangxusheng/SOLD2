#!/bin/bash

# Step 5: Train the full pipeline on the Wireframe dataset
# You first need to modify the field 'return_type' in config/wireframe_dataset.yaml to 'paired_desc'.
python -m sold2.experiment --mode train --dataset_config sold2/config/merge_dataset_mini_desc.yaml \
--model_config sold2/config/train_full_pipeline_mini.yaml --exp_name sold2_synth_superpoint_128x128_ft1_full \
--pretrained --pretrained_path experiments/sold2_synth_superpoint_128x128_ft1 --checkpoint_name checkpoint-epoch110-end.tar