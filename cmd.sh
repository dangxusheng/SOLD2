

cd /home/sunnypc/dangxs/projects/python_projects/SOLD2-main && \
conda activate py37_torch180_cu113


#Step 1: Train on a synthetic dataset
python -m sold2.experiment --mode train --dataset_config sold2/config/synthetic_dataset.yaml \
--model_config sold2/config/train_detector.yaml --exp_name sold2_synth

#Step 2: Export the raw pseudo ground truth on the Wireframe dataset with homography adaptation
python -m sold2.experiment --exp_name wireframe_train --mode export --resume_path <path to your previously trained sold2_synth> \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/wireframe_dataset.yaml \
--checkpoint_name <name of the best checkpoint> --export_dataset_mode train --export_batch_size 4

#Step3: Compute the ground truth line segments from the raw data
python -m sold2.postprocess.convert_homography_results <name of the previously exported file (e.g. "wireframe_train.h5")> <name of the new data with extracted line segments (e.g. "wireframe_train_gt.h5")> sold2/config/export_line_features.yaml

#Step 4: Train the detector on the Wireframe dataset
python -m sold2.experiment --mode train --dataset_config sold2/config/wireframe_dataset.yaml \
--model_config sold2/config/train_detector.yaml --exp_name sold2_wireframe

python -m sold2.experiment --mode train --dataset_config sold2/config/wireframe_dataset.yaml \
--model_config sold2/config/train_detector.yaml --exp_name sold2_wireframe \
--pretrained --pretrained_path <path ot the pre-trained sold2_synth> --checkpoint_name <name of the best checkpoint>


#Step 5: Train the full pipeline on the Wireframe dataset
python -m sold2.experiment --mode train --dataset_config sold2/config/wireframe_dataset.yaml \
--model_config sold2/config/train_full_pipeline.yaml --exp_name sold2_full_wireframe \
--pretrained --pretrained_path <path ot the pre-trained sold2_wireframe> --checkpoint_name <name of the best checkpoint>

######################################################################################

# step1
python -m sold2.experiment --mode train --dataset_config sold2/config/synthetic_dataset_mini.yaml \
--resume_path experiments/sold2_synth_superpoint --checkpoint_name checkpoint-epoch000-end.tar \
--model_config sold2/config/train_detector_mini.yaml --exp_name sold2_synth_superpoint


python -m sold2.experiment --mode train --dataset_config sold2/config/synthetic_dataset_mini.yaml \
--pretrained --resume_path experiments/sold2_synth_superpoint --checkpoint_name checkpoint-epoch000-end.tar \
--model_config sold2/config/train_detector_mini.yaml --exp_name sold2_synth_superpoint


# step 2
python -m sold2.experiment --exp_name tray_train --mode export --resume_path pretrained_models \
--model_config sold2/config/train_detector.yaml --dataset_config sold2/config/tray_dataset.yaml \
--checkpoint_name sold2_wireframe.tar --export_dataset_mode train --export_batch_size 4

#step3
python -m sold2.postprocess.convert_homography_results "tray_train.h5" "tray_train_gt.h5" sold2/config/export_line_features2.yaml


# step4
python -m sold2.experiment --mode train --dataset_config sold2/config/tray_dataset_train.yaml \
--model_config sold2/config/train_detector.yaml --exp_name sold2_tray \
--pretrained --pretrained_path pretrained_models --checkpoint_name sold2_wireframe.tar


python -m sold2.experiment --mode train --dataset_config sold2/config/tray_dataset_train.yaml \
--model_config sold2/config/train_detector.yaml --exp_name sold2_tray \
--pretrained --pretrained_path pretrained_models --checkpoint_name sold2_wireframe.tar



# 后台运行
nohup sh ./step1_train_synthetic.sh > run_nohup.log 2>&1 &