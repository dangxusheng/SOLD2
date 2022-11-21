#!/bin/bash

# Step3: Compute the ground truth line segments from the raw data

python -m sold2.postprocess.convert_homography_results "sold2_synth_wireframe_train.h5" "sold2_synth_wireframe_train_gt.h5" \
sold2/config/export_line_features.yaml
python -m sold2.postprocess.convert_homography_results "sold2_synth_wireframe_test.h5" "sold2_synth_wireframe_test_gt.h5" \
sold2/config/export_line_features.yaml

python -m sold2.postprocess.convert_homography_results "sold2_synth_yorkUrban_train.h5" "sold2_synth_yorkUrban_train_gt.h5" \
sold2/config/export_line_features.yaml
python -m sold2.postprocess.convert_homography_results "sold2_synth_yorkUrban_test.h5" "sold2_synth_yorkUrban_test_gt.h5" \
sold2/config/export_line_features.yaml

python -m sold2.postprocess.convert_homography_results "sold2_synth_tray_train.h5" "sold2_synth_tray_train_gt.h5" \
sold2/config/export_line_features.yaml


###TODO:::  with pair_desc

python -m sold2.postprocess.convert_homography_results "sold2_synth_wireframe_train_desc.h5" "sold2_synth_wireframe_train_desc_gt.h5" \
sold2/config/export_line_features.yaml
python -m sold2.postprocess.convert_homography_results "sold2_synth_wireframe_test_desc.h5" "sold2_synth_wireframe_test_desc_gt.h5" \
sold2/config/export_line_features.yaml

python -m sold2.postprocess.convert_homography_results "sold2_synth_yorkUrban_train_desc.h5" "sold2_synth_yorkUrban_train_desc_gt.h5" \
sold2/config/export_line_features.yaml

python -m sold2.postprocess.convert_homography_results "sold2_synth_tray_train_desc.h5" "sold2_synth_tray_train_desc_gt.h5" \
sold2/config/export_line_features.yaml