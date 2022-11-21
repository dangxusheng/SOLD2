
# step1:  Train on a synthetic dataset
python -m sold2.experiment --mode train --dataset_config sold2/config/synthetic_dataset_mini.yaml \
--resume --resume_path experiments/sold2_synth_superpoint --checkpoint_name checkpoint-epoch110-end.tar \
--model_config sold2/config/train_detector_mini2.yaml --exp_name sold2_synth_superpoint_ft1


python -m sold2.experiment --mode train --dataset_config sold2/config/synthetic_dataset_mini.yaml \
--model_config sold2/config/train_detector_mini.yaml --exp_name sold2_synth_superpoint_128x128


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


# Step3: Compute the ground truth line segments from the raw data

python -m sold2.postprocess.convert_homography_results "sold2_synth_wireframe_train.h5" "sold2_synth_wireframe_train_gt.h5" \
sold2/config/export_line_features_mini.yaml
python -m sold2.postprocess.convert_homography_results "sold2_synth_wireframe_test.h5" "sold2_synth_wireframe_test_gt.h5" \
sold2/config/export_line_features_mini.yaml

python -m sold2.postprocess.convert_homography_results "sold2_synth_yorkUrban_train.h5" "sold2_synth_yorkUrban_train_gt.h5" \
sold2/config/export_line_features_mini.yaml
python -m sold2.postprocess.convert_homography_results "sold2_synth_tray_train.h5" "sold2_synth_tray_train_gt.h5" \
sold2/config/export_line_features_mini.yaml



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
--model_config sold2/config/train_detector_mini.yaml --exp_name sold2_merge \
--pretrained --pretrained_path experiments/sold2_synth_superpoint --checkpoint_name checkpoint-epoch105-end.tar


# Step 5: Train the full pipeline on the Wireframe dataset
# You first need to modify the field 'return_type' in config/wireframe_dataset.yaml to 'paired_desc'.
python -m sold2.experiment --mode train --dataset_config sold2/config/wireframe_dataset_no_gt_mini.yaml \
--model_config sold2/config/train_full_pipeline.yaml --exp_name sold2_full_wireframe \
--pretrained --pretrained_path experiments/sold2_wireframe --checkpoint_name xxxx.tar