#!/bin/bash

# Step 2: Export the raw pseudo ground truth on the Wireframe dataset with homography adaptation
### wireframe
python -m sold2.experiment --exp_name sold2_synth_wireframe_train --mode export --resume_path pretrained_models \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/wireframe_dataset_no_gt_mini.yaml \
--checkpoint_name sold2_wireframe.tar --export_dataset_mode train --export_batch_size 8

python -m sold2.experiment --exp_name sold2_synth_wireframe_test --mode export --resume_path pretrained_models \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/wireframe_dataset_no_gt_mini.yaml \
--checkpoint_name sold2_wireframe.tar --export_dataset_mode test --export_batch_size 8

### yorkUrban
python -m sold2.experiment --exp_name sold2_synth_yorkUrban_train --mode export --resume_path pretrained_models \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/yorkUrban_dataset_no_gt_mini.yaml \
--checkpoint_name sold2_wireframe.tar --export_dataset_mode train --export_batch_size 8

python -m sold2.experiment --exp_name sold2_synth_yorkUrban_test --mode export --resume_path pretrained_models \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/yorkUrban_dataset_no_gt_mini.yaml \
--checkpoint_name sold2_wireframe.tar --export_dataset_mode test --export_batch_size 8

### tray
python -m sold2.experiment --exp_name sold2_synth_tray_train --mode export --resume_path pretrained_models \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/tray_dataset.yaml \
--checkpoint_name sold2_wireframe.tar --export_dataset_mode train --export_batch_size 4


###TODO:::  with pair_desc

### wireframe
python -m sold2.experiment --exp_name sold2_synth_wireframe_train_desc --mode export --resume_path pretrained_models \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/wireframe_dataset_no_gt_mini_desc.yaml \
--checkpoint_name sold2_wireframe.tar --export_dataset_mode train --export_batch_size 8

python -m sold2.experiment --exp_name sold2_synth_wireframe_test_desc --mode export --resume_path pretrained_models \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/wireframe_dataset_no_gt_mini_desc.yaml \
--checkpoint_name sold2_wireframe.tar --export_dataset_mode test --export_batch_size 8

### yorkUrban
python -m sold2.experiment --exp_name sold2_synth_yorkUrban_train_desc --mode export --resume_path pretrained_models \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/yorkUrban_dataset_no_gt_mini_desc.yaml \
--checkpoint_name sold2_wireframe.tar --export_dataset_mode train --export_batch_size 8

python -m sold2.experiment --exp_name sold2_synth_yorkUrban_test_desc --mode export --resume_path pretrained_models \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/yorkUrban_dataset_no_gt_mini_desc.yaml \
--checkpoint_name sold2_wireframe.tar --export_dataset_mode test --export_batch_size 8

### tray
python -m sold2.experiment --exp_name sold2_synth_tray_train_desc --mode export --resume_path pretrained_models \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/tray_dataset_desc.yaml \
--checkpoint_name sold2_wireframe.tar --export_dataset_mode train --export_batch_size 4