#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: Jory.d
@contact: 707564875@qq.com
@site: 
@software: PyCharm
@file: ncnn_test.py
@time: 2022/11/16 上午10:51
@desc:
"""

"""
实现sold2模型的推理+后处理，端到端

python ncnn_test.py

网络前向耗时：0.025s
后处理： 1.9s
在pc上推理都要 2s

"""

import cv2
import math
import numpy as np
from ncnn_basenet import NCNNBaseNet
import time

# overwrite torch function
class torch:
    int = np.int32
    long = np.int64
    @classmethod
    def cat(cls, datas, dim=-1):
        return np.concatenate(datas, axis=dim)

    @classmethod
    def clamp(cls, data, min, max):
        return np.clip(data, min, max)

    @classmethod
    def mean(cls, data, dim=-1):
        return np.mean(data, axis=dim)

    @classmethod
    def sum(cls, data, dim):
        return np.sum(data, axis=dim)

    @classmethod
    def argmax(cls, data, dim=-1):
        return np.argmax(data, axis=dim)

    @classmethod
    def unsqueeze(cls, data, dim=-1):
        return np.expand_dims(data, axis=dim)

    @classmethod
    def where(cls, data):
        return np.where(data)

    @classmethod
    def triu(cls, data, diagonal=1):
        return np.triu(data, k=diagonal)

    @classmethod
    def norm(cls, data, dim=1):
        return np.linalg.norm(data, axis=dim)

    @classmethod
    def acos(cls, data):
        return np.arccos(data)

    @classmethod
    def sin(cls, data):
        return np.sin(data)

    @classmethod
    def arange(cls, start, end, step=1):
        return np.arange(start, end, step)

    @classmethod
    def meshgrid(cls, *xi, **kwargs):
        return np.meshgrid(*xi, kwargs)

    @classmethod
    def sqrt(cls, data):
        return np.sqrt(data)

    @classmethod
    def unique(cls, data, dim=-1):
        return np.unique(data, axis=dim)


class LineSegmentDetectionModule(object):
    """ Module extracting line segments from junctions and line heatmaps. """

    def __init__(
            self, detect_thresh, num_samples=64, sampling_method="local_max",
            inlier_thresh=0., heatmap_low_thresh=0.15, heatmap_high_thresh=0.2,
            max_local_patch_radius=3, lambda_radius=2.,
            use_candidate_suppression=False, nms_dist_tolerance=3.,
            use_heatmap_refinement=False, heatmap_refine_cfg=None,
            use_junction_refinement=False, junction_refine_cfg=None):
        """
        Parameters:
            detect_thresh: The probability threshold for mean activation (0. ~ 1.)
            num_samples: Number of sampling locations along the line segments.
            sampling_method: Sampling method on locations ("bilinear" or "local_max").
            inlier_thresh: The min inlier ratio to satisfy (0. ~ 1.) => 0. means no threshold.
            heatmap_low_thresh: The lowest threshold for the pixel to be considered as candidate in junction recovery.
            heatmap_high_thresh: The higher threshold for NMS in junction recovery.
            max_local_patch_radius: The max patch to be considered in local maximum search.
            lambda_radius: The lambda factor in linear local maximum search formulation
            use_candidate_suppression: Apply candidate suppression to break long segments into short sub-segments.
            nms_dist_tolerance: The distance tolerance for nms. Decide whether the junctions are on the line.
            use_heatmap_refinement: Use heatmap refinement method or not.
            heatmap_refine_cfg: The configs for heatmap refinement methods.
            use_junction_refinement: Use junction refinement method or not.
            junction_refine_cfg: The configs for junction refinement methods.
        """
        # Line detection parameters
        self.detect_thresh = detect_thresh

        # Line sampling parameters
        self.num_samples = num_samples
        self.sampling_method = sampling_method
        self.inlier_thresh = inlier_thresh
        self.local_patch_radius = max_local_patch_radius
        self.lambda_radius = lambda_radius

        # Detecting junctions on the boundary parameters
        self.low_thresh = heatmap_low_thresh
        self.high_thresh = heatmap_high_thresh

        # Pre-compute the linspace sampler
        self.sampler = np.linspace(0, 1, self.num_samples)

        # Long line segment suppression configuration
        self.use_candidate_suppression = use_candidate_suppression
        self.nms_dist_tolerance = nms_dist_tolerance

        # Heatmap refinement configuration
        self.use_heatmap_refinement = use_heatmap_refinement
        self.heatmap_refine_cfg = heatmap_refine_cfg
        if self.use_heatmap_refinement and self.heatmap_refine_cfg is None:
            raise ValueError("[Error] Missing heatmap refinement config.")

        # Junction refinement configuration
        self.use_junction_refinement = use_junction_refinement
        self.junction_refine_cfg = junction_refine_cfg
        if self.use_junction_refinement and self.junction_refine_cfg is None:
            raise ValueError("[Error] Missing junction refinement config.")

    def detect(self, junctions, heatmap):
        """ Main function performing line segment detection. """
        # Perform the heatmap refinement
        if self.use_heatmap_refinement:
            if self.heatmap_refine_cfg["mode"] == "global":
                heatmap = self.refine_heatmap(
                    heatmap,
                    self.heatmap_refine_cfg["ratio"],
                    self.heatmap_refine_cfg["valid_thresh"]
                )
            elif self.heatmap_refine_cfg["mode"] == "local":
                heatmap = self.refine_heatmap_local(
                    heatmap,
                    self.heatmap_refine_cfg["num_blocks"],
                    self.heatmap_refine_cfg["overlap_ratio"],
                    self.heatmap_refine_cfg["ratio"],
                    self.heatmap_refine_cfg["valid_thresh"]
                )

        # Initialize empty line map
        num_junctions = junctions.shape[0]
        line_map_pred = np.zeros([num_junctions, num_junctions], dtype=np.int32)

        # Stop if there are not enough junctions
        if num_junctions < 2:
            return line_map_pred, junctions, heatmap

        # Generate the candidate map
        candidate_map = np.triu(np.ones(
            [num_junctions, num_junctions], dtype=np.int32),
            k=1)

        # Fetch the image boundary
        if len(heatmap.shape) > 2:
            H, W, _ = heatmap.shape
        else:
            H, W = heatmap.shape

        # Optionally perform candidate filtering
        if self.use_candidate_suppression:
            candidate_map = self.candidate_suppression(junctions,
                                                       candidate_map)

        # Fetch the candidates
        candidate_index_map = torch.where(candidate_map)
        candidate_index_map = torch.cat([candidate_index_map[0][..., None],
                                         candidate_index_map[1][..., None]],
                                        dim=-1)

        # Get the corresponding start and end junctions
        candidate_junc_start = junctions[candidate_index_map[:, 0], :]
        candidate_junc_end = junctions[candidate_index_map[:, 1], :]

        # Get the sampling locations (N x 64)
        sampler = self.sampler[None, ...]
        cand_samples_h = candidate_junc_start[:, 0:1] * sampler + \
                         candidate_junc_end[:, 0:1] * (1 - sampler)
        cand_samples_w = candidate_junc_start[:, 1:2] * sampler + \
                         candidate_junc_end[:, 1:2] * (1 - sampler)

        # Clip to image boundary
        cand_h = torch.clamp(cand_samples_h, min=0, max=H - 1)
        cand_w = torch.clamp(cand_samples_w, min=0, max=W - 1)

        # Local maximum search
        if self.sampling_method == "local_max":
            # Compute normalized segment lengths
            segments_length = torch.sqrt(torch.sum(
                (candidate_junc_start - candidate_junc_end) ** 2, dim=-1))
            normalized_seg_length = (segments_length
                                     / (((H ** 2) + (W ** 2)) ** 0.5))

            # Perform local max search
            num_cand = cand_h.shape[0]
            group_size = 10000
            if num_cand > group_size:
                num_iter = math.ceil(num_cand / group_size)
                sampled_feat_lst = []
                for iter_idx in range(num_iter):
                    if not iter_idx == num_iter - 1:
                        cand_h_ = cand_h[iter_idx * group_size:
                                         (iter_idx + 1) * group_size, :]
                        cand_w_ = cand_w[iter_idx * group_size:
                                         (iter_idx + 1) * group_size, :]
                        normalized_seg_length_ = normalized_seg_length[
                                                 iter_idx * group_size: (iter_idx + 1) * group_size]
                    else:
                        cand_h_ = cand_h[iter_idx * group_size:, :]
                        cand_w_ = cand_w[iter_idx * group_size:, :]
                        normalized_seg_length_ = normalized_seg_length[
                                                 iter_idx * group_size:]
                    sampled_feat_ = self.detect_local_max(
                        heatmap, cand_h_, cand_w_, H, W,
                        normalized_seg_length_)
                    sampled_feat_lst.append(sampled_feat_)
                sampled_feat = np.concatenate(sampled_feat_lst, axis=0)
            else:
                sampled_feat = self.detect_local_max(
                    heatmap, cand_h, cand_w, H, W,
                    normalized_seg_length)
        # Bilinear sampling
        elif self.sampling_method == "bilinear":
            # Perform bilinear sampling
            sampled_feat = self.detect_bilinear(
                heatmap, cand_h, cand_w, H, W)
        else:
            raise ValueError("[Error] Unknown sampling method.")

        # [Simple threshold detection]
        # detection_results is a mask over all candidates
        detection_results = (np.mean(sampled_feat, axis=-1)
                             > self.detect_thresh)

        # [Inlier threshold detection]
        if self.inlier_thresh > 0.:
            inlier_ratio = torch.sum(
                sampled_feat > self.detect_thresh,
                dim=-1) / self.num_samples
            detection_results_inlier = inlier_ratio >= self.inlier_thresh
            detection_results = detection_results * detection_results_inlier

        # Convert detection results back to line_map_pred
        detected_junc_indexes = candidate_index_map[detection_results, :]
        line_map_pred[detected_junc_indexes[:, 0],
                      detected_junc_indexes[:, 1]] = 1
        line_map_pred[detected_junc_indexes[:, 1],
                      detected_junc_indexes[:, 0]] = 1

        # Perform junction refinement
        if self.use_junction_refinement and len(detected_junc_indexes) > 0:
            junctions, line_map_pred = self.refine_junction_perturb(
                junctions, line_map_pred, heatmap, H, W)

        return line_map_pred, junctions, heatmap

    def refine_heatmap(self, heatmap, ratio=0.2, valid_thresh=1e-2):
        """ Global heatmap refinement method. """
        # Grab the top 10% values
        heatmap_values = heatmap[heatmap > valid_thresh]
        # sorted_values = torch.sort(heatmap_values, descending=True)[0]
        sorted_values = np.sort(heatmap_values)[::-1]
        top10_len = math.ceil(sorted_values.shape[0] * ratio)
        max20 = np.mean(sorted_values[:top10_len])
        heatmap = torch.clamp(heatmap / max20, min=0., max=1.)
        return heatmap

    def refine_heatmap_local(self, heatmap, num_blocks=5, overlap_ratio=0.5,
                             ratio=0.2, valid_thresh=2e-3):
        """ Local heatmap refinement method. """
        # Get the shape of the heatmap
        H, W = heatmap.shape
        increase_ratio = 1 - overlap_ratio
        h_block = round(H / (1 + (num_blocks - 1) * increase_ratio))
        w_block = round(W / (1 + (num_blocks - 1) * increase_ratio))

        count_map = np.zeros(heatmap.shape, dtype=np.int32)
        heatmap_output = np.zeros(heatmap.shape, dtype=np.float32)
        # Iterate through each block
        for h_idx in range(num_blocks):
            for w_idx in range(num_blocks):
                # Fetch the heatmap
                h_start = round(h_idx * h_block * increase_ratio)
                w_start = round(w_idx * w_block * increase_ratio)
                h_end = h_start + h_block if h_idx < num_blocks - 1 else H
                w_end = w_start + w_block if w_idx < num_blocks - 1 else W

                subheatmap = heatmap[h_start:h_end, w_start:w_end]
                if subheatmap.max() > valid_thresh:
                    subheatmap = self.refine_heatmap(
                        subheatmap, ratio, valid_thresh=valid_thresh)

                # Aggregate it to the final heatmap
                heatmap_output[h_start:h_end, w_start:w_end] += subheatmap
                count_map[h_start:h_end, w_start:w_end] += 1
        heatmap_output = torch.clamp(heatmap_output / count_map, max=1., min=0.)

        return heatmap_output

    def candidate_suppression(self, junctions, candidate_map):
        """ Suppress overlapping long lines in the candidate segments. """
        # Define the distance tolerance
        dist_tolerance = self.nms_dist_tolerance

        # Compute distance between junction pairs
        # (num_junc x 1 x 2) - (1 x num_junc x 2) => num_junc x num_junc map
        line_dist_map = torch.sum((torch.unsqueeze(junctions, dim=1)
                                   - junctions[None, ...]) ** 2, dim=-1) ** 0.5

        # Fetch all the "detected lines"
        seg_indexes = torch.where(torch.triu(candidate_map, diagonal=1))
        start_point_idxs = seg_indexes[0]
        end_point_idxs = seg_indexes[1]
        start_points = junctions[start_point_idxs, :]
        end_points = junctions[end_point_idxs, :]

        # Fetch corresponding entries
        line_dists = line_dist_map[start_point_idxs, end_point_idxs]

        # Check whether they are on the line
        dir_vecs = ((end_points - start_points)
                    / torch.norm(end_points - start_points,
                                 dim=-1)[..., None])
        # Get the orthogonal distance
        # cand_vecs = junctions[None, ...] - start_points.unsqueeze(dim=1)
        cand_vecs = junctions[None, ...] - np.expand_dims(start_points, axis=1)
        cand_vecs_norm = torch.norm(cand_vecs, dim=-1)
        # Check whether they are projected directly onto the segment
        proj = (np.einsum('bij,bjk->bik', cand_vecs, dir_vecs[..., None])
                / line_dists[..., None, None])
        # proj is num_segs x num_junction x 1
        proj_mask = (proj >= 0) * (proj <= 1)
        cand_angles = torch.acos(
            np.einsum('bij,bjk->bik', cand_vecs, dir_vecs[..., None])
            / cand_vecs_norm[..., None])
        cand_dists = cand_vecs_norm[..., None] * torch.sin(cand_angles)
        junc_dist_mask = cand_dists <= dist_tolerance
        junc_mask = junc_dist_mask * proj_mask

        # Minus starting points
        num_segs = start_point_idxs.shape[0]
        # junc_counts = torch.sum(junc_mask, dim=[1, 2])
        junc_counts = torch.sum(junc_mask, dim=(1, 2))
        junc_counts -= junc_mask[..., 0][torch.arange(0, num_segs), start_point_idxs]
        junc_counts -= junc_mask[..., 0][torch.arange(0, num_segs),
                                         end_point_idxs]

        # Get the invalid candidate mask
        final_mask = junc_counts > 0
        candidate_map[start_point_idxs[final_mask],
                      end_point_idxs[final_mask]] = 0

        return candidate_map

    def refine_junction_perturb(self, junctions, line_map_pred,
                                heatmap, H, W):
        """ Refine the line endpoints in a similar way as in LSD. """
        # Get the config
        junction_refine_cfg = self.junction_refine_cfg

        # Fetch refinement parameters
        num_perturbs = junction_refine_cfg["num_perturbs"]
        perturb_interval = junction_refine_cfg["perturb_interval"]
        side_perturbs = (num_perturbs - 1) // 2
        # Fetch the 2D perturb mat
        perturb_vec = torch.arange(
            start=-perturb_interval * side_perturbs,
            end=perturb_interval * (side_perturbs + 1),
            step=perturb_interval)

        w1_grid, h1_grid, w2_grid, h2_grid = np.meshgrid(
            perturb_vec, perturb_vec, perturb_vec, perturb_vec)
        perturb_tensor = torch.cat([
            w1_grid[..., None], h1_grid[..., None],
            w2_grid[..., None], h2_grid[..., None]], dim=-1)
        # perturb_tensor_flat = perturb_tensor.view(-1, 2, 2)
        perturb_tensor_flat = perturb_tensor.reshape(-1, 2, 2)

        # Fetch the junctions and line_map
        # junctions = junctions.clone()
        junctions = np.array(junctions)
        line_map = line_map_pred

        # Fetch all the detected lines
        detected_seg_indexes = torch.where(torch.triu(line_map, diagonal=1))
        start_point_idxs = detected_seg_indexes[0]
        end_point_idxs = detected_seg_indexes[1]
        start_points = junctions[start_point_idxs, :]
        end_points = junctions[end_point_idxs, :]

        start_points = np.expand_dims(start_points, axis=1)
        end_points = np.expand_dims(end_points, axis=1)

        line_segments = torch.cat([start_points, end_points], dim=1)
        line_segments = np.expand_dims(line_segments, axis=1)
        line_segment_candidates = (line_segments + perturb_tensor_flat[None, ...])
        # Clip the boundaries
        line_segment_candidates[..., 0] = torch.clamp(
            line_segment_candidates[..., 0], min=0, max=H - 1)
        line_segment_candidates[..., 1] = torch.clamp(
            line_segment_candidates[..., 1], min=0, max=W - 1)

        # Iterate through all the segments
        refined_segment_lst = []
        num_segments = line_segments.shape[0]
        for idx in range(num_segments):
            segment = line_segment_candidates[idx, ...]
            # Get the corresponding start and end junctions
            candidate_junc_start = segment[:, 0, :]
            candidate_junc_end = segment[:, 1, :]

            # Get the sampling locations (N x 64)
            sampler = self.sampler[None, ...]
            cand_samples_h = (candidate_junc_start[:, 0:1] * sampler +
                              candidate_junc_end[:, 0:1] * (1 - sampler))
            cand_samples_w = (candidate_junc_start[:, 1:2] * sampler +
                              candidate_junc_end[:, 1:2] * (1 - sampler))

            # Clip to image boundary
            cand_h = torch.clamp(cand_samples_h, min=0, max=H - 1)
            cand_w = torch.clamp(cand_samples_w, min=0, max=W - 1)

            # Perform bilinear sampling
            segment_feat = self.detect_bilinear(
                heatmap, cand_h, cand_w, H, W)
            segment_results = torch.mean(segment_feat, dim=-1)
            max_idx = torch.argmax(segment_results)
            refined_segment_lst.append(segment[max_idx, ...][None, ...])

        # Concatenate back to segments
        refined_segments = torch.cat(refined_segment_lst, dim=0)

        # Convert back to junctions and line_map
        junctions_new = torch.cat(
            [refined_segments[:, 0, :], refined_segments[:, 1, :]], dim=0)
        junctions_new = torch.unique(junctions_new, dim=0)
        line_map_new = self.segments_to_line_map(junctions_new,
                                                 refined_segments)

        return junctions_new, line_map_new

    def segments_to_line_map(self, junctions, segments):
        """ Convert the list of segments to line map. """
        # Create empty line map
        num_junctions = junctions.shape[0]
        line_map = np.zeros([num_junctions, num_junctions])

        # Iterate through every segment
        for idx in range(segments.shape[0]):
            # Get the junctions from a single segement
            seg = segments[idx, ...]
            junction1 = seg[0, :]
            junction2 = seg[1, :]
            # Get index
            idx_junction1 = np.where(
                np.sum(junctions == junction1, axis=1) == 2)[0]

            idx_junction2 = np.where(
                np.sum(junctions == junction2, axis=1) == 2)[0]

            # label the corresponding entries
            line_map[idx_junction1, idx_junction2] = 1
            line_map[idx_junction2, idx_junction1] = 1

        return line_map

    def detect_bilinear(self, heatmap, cand_h, cand_w, H, W):
        """ Detection by bilinear sampling. """
        # Get the floor and ceiling locations
        cand_h_floor = np.floor(cand_h).astype(np.int64)
        cand_h_ceil = np.ceil(cand_h).astype(np.int64)
        cand_w_floor = np.floor(cand_w).astype(np.int64)
        cand_w_ceil = np.ceil(cand_w).astype(np.int64)

        # Perform the bilinear sampling
        cand_samples_feat = (
                heatmap[cand_h_floor, cand_w_floor] * (cand_h_ceil - cand_h)
                * (cand_w_ceil - cand_w) + heatmap[cand_h_floor, cand_w_ceil]
                * (cand_h_ceil - cand_h) * (cand_w - cand_w_floor) +
                heatmap[cand_h_ceil, cand_w_floor] * (cand_h - cand_h_floor)
                * (cand_w_ceil - cand_w) + heatmap[cand_h_ceil, cand_w_ceil]
                * (cand_h - cand_h_floor) * (cand_w - cand_w_floor))

        return cand_samples_feat

    def detect_local_max(self, heatmap, cand_h, cand_w, H, W,
                         normalized_seg_length):
        """ Detection by local maximum search. """
        # Compute the distance threshold
        dist_thresh = (0.5 * (2 ** 0.5)
                       + self.lambda_radius * normalized_seg_length)
        # Make it N x 64
        # dist_thresh = torch.repeat_interleave(dist_thresh[..., None],self.num_samples, dim=-1)
        dist_thresh = np.repeat(dist_thresh[..., None], self.num_samples, axis=-1)

        # Compute the candidate points
        cand_points = torch.cat([cand_h[..., None], cand_w[..., None]],
                                dim=-1)
        cand_points_round = np.round(cand_points)  # N x 64 x 2

        # Construct local patches 9x9 = 81
        patch_mask = np.zeros([int(2 * self.local_patch_radius + 1),
                               int(2 * self.local_patch_radius + 1)])
        patch_center = np.array(
            [[self.local_patch_radius, self.local_patch_radius]], dtype=np.float32)
        H_patch_points, W_patch_points = torch.where(patch_mask >= 0)
        patch_points = torch.cat([H_patch_points[..., None],
                                  W_patch_points[..., None]], dim=-1)
        # Fetch the circle region
        patch_center_dist = torch.sqrt(torch.sum(
            (patch_points - patch_center) ** 2, dim=-1))
        patch_points = (patch_points[patch_center_dist
                                     <= self.local_patch_radius, :])
        # Shift [0, 0] to the center
        patch_points = patch_points - self.local_patch_radius

        # Construct local patch mask
        patch_points_shifted = (torch.unsqueeze(cand_points_round, dim=2)
                                + patch_points[None, None, ...])
        patch_dist = torch.sqrt(torch.sum((torch.unsqueeze(cand_points, dim=2)
                                           - patch_points_shifted) ** 2,
                                          dim=-1))
        patch_dist_mask = patch_dist < dist_thresh[..., None]

        # Get all points => num_points_center x num_patch_points x 2
        points_H = torch.clamp(patch_points_shifted[:, :, :, 0], min=0,
                               max=H - 1)
        points_W = torch.clamp(patch_points_shifted[:, :, :, 1], min=0,
                               max=W - 1)
        points = torch.cat([points_H[..., None], points_W[..., None]], dim=-1)

        points = np.asarray(points, dtype=np.int32)
        # Sample the feature (N x 64 x 81)
        sampled_feat = heatmap[points[:, :, :, 0], points[:, :, :, 1]]
        # Filtering using the valid mask
        sampled_feat = sampled_feat * patch_dist_mask
        if len(sampled_feat) == 0:
            sampled_feat_lmax = np.empty(0, 64)
        else:
            # sampled_feat_lmax, _ = torch.max(sampled_feat, dim=-1)
            sampled_feat_lmax = np.max(sampled_feat, axis=-1)

        return sampled_feat_lmax


class NCNN_LineDetectNet(NCNNBaseNet):
    CLASSES = ('junction', '__backgound__',)

    MODEL_ROOT = 'experiments/sold2_synth_superpoint_128x128_ft1_full'
    PARAM_PATH = f'{MODEL_ROOT}/checkpoint-epoch085-end.tar.onnx.opt.param'
    BIN_PATH = f'{MODEL_ROOT}/checkpoint-epoch085-end.tar.onnx.opt.bin'
    # PARAM_PATH = f'{MODEL_ROOT}/checkpoint-epoch085-end.tar.onnx.opt.fp16.param'
    # BIN_PATH = f'{MODEL_ROOT}/checkpoint-epoch085-end.tar.onnx.opt.fp16.bin'

    INPUT_W = 320
    INPUT_H = 320
    INPUT_C = 1
    MEAN = [0., ] * INPUT_C
    STD = [1 / 255., ] * INPUT_C

    OUTPUT_NODES = [
        '72',  # junctions      [65,16,16]
        '85',  # heatmap        [2, INPUT_H,INPUT_W]
        '88'  # descriptor      [64,16,16]
    ]

    def __init__(self):
        super().__init__()

        #####################################################################
        # params:  from sold2/config/export_line_features_mini.yaml
        self.grid_size = 8
        self.keep_border_valid = True
        self.junc_detect_thresh = 0.0153846  # 1/65
        self.max_num_junctions = 300
        # Threshold of heatmap detection
        self.prob_thresh = 0.5

        line_detector_cfg = {
            "detect_thresh": 0.8,
            "num_samples": 64,
            "sampling_method": "local_max",
            "inlier_thresh": 0.98,
            "use_candidate_suppression": True,
            "nms_dist_tolerance": 2.,
            "use_heatmap_refinement": True,
            "heatmap_refine_cfg":
                {
                    "mode": "local",
                    "ratio": 0.2,
                    "valid_thresh": 0.001,
                    "num_blocks": 20,
                    "overlap_ratio": 0.5,
                },
            "use_junction_refinement": True,
            "junction_refine_cfg": {
                "num_perturbs": 9,
                "perturb_interval": 0.25,
            }
        }
        #####################################################################
        self.line_detector = LineSegmentDetectionModule(**line_detector_cfg)

    def pixel_shuffle(self, junc_pred, grid_size=8):
        """
        pytorch实现： https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
        Rearranges elements in a tensor of shape (*, C x r^2, H, W)
        to a tensor of shape (*, C, Hxr, Wxr), where r is an upscale factor.

        使用numpy来实现 torch的pixel_shuffle()

        """
        r = grid_size
        if len(junc_pred.shape) == 3:
            junc_pred = junc_pred[None, ...]  # [c,h,w] -- > [b,c,h,w]
        b, c, h, w = junc_pred.shape
        c1 = c // (r ** 2)
        junc_pred = np.reshape(junc_pred, [b, c1, r ** 2, h, w])
        junc_pred = np.transpose(junc_pred, axes=[0, 1, 3, 4, 2])
        junc_pred = np.reshape(junc_pred, [b, c1, h, w, r, r])
        junc_pred = np.transpose(junc_pred, axes=[0, 1, 2, 4, 3, 5])
        junc_pred = np.reshape(junc_pred, [b, c1, h * r, w * r])
        return junc_pred

    # def tets_pixel_shuffle(self):
    #     import torch
    #     import torch.nn.functional as F
    #     a = torch.randn(1, 8, 2, 2)
    #     a_np = a.numpy()
    #     a1 = F.pixel_shuffle(a, 2).numpy()
    #     a2 = self.pixel_shuffle(a_np, 2)
    #     print(a1.shape)
    #     print(a2.shape)
    #     print('a1: ', a1)
    #     print('a2: ', a2)
    #     print(np.array_equal(a1, a2))

    def super_nms(self, prob_predictions, dist_thresh, prob_thresh=0.01, top_k=0):
        """ Non-maximum suppression adapted from SuperPoint. """
        # Iterate through batch dimension
        im_h = prob_predictions.shape[1]
        im_w = prob_predictions.shape[2]
        output_lst = []
        for i in range(prob_predictions.shape[0]):
            # print(i)
            prob_pred = prob_predictions[i, ...]
            # Filter the points using prob_thresh
            coord = np.where(prob_pred >= prob_thresh)  # HW format
            points = np.concatenate((coord[0][..., None], coord[1][..., None]),
                                    axis=1)  # HW format

            # Get the probability score
            prob_score = prob_pred[points[:, 0], points[:, 1]]

            # Perform super nms
            # Modify the in_points to xy format (instead of HW format)
            in_points = np.concatenate((coord[1][..., None], coord[0][..., None],
                                        prob_score), axis=1).T
            keep_points_, keep_inds = self.nms_fast(in_points, im_h, im_w, dist_thresh)
            # Remember to flip outputs back to HW format
            keep_points = np.round(np.flip(keep_points_[:2, :], axis=0).T)
            keep_score = keep_points_[-1, :].T

            # Whether we only keep the topk value
            if (top_k > 0) or (top_k is None):
                k = min([keep_points.shape[0], top_k])
                keep_points = keep_points[:k, :]
                keep_score = keep_score[:k]

            # Re-compose the probability map
            output_map = np.zeros([im_h, im_w])
            output_map[keep_points[:, 0].astype(np.int),
                       keep_points[:, 1].astype(np.int)] = keep_score.squeeze()

            output_lst.append(output_map[None, ...])

        return np.concatenate(output_lst, axis=0)

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1,
        rest are zeros. Iterate through all the 1's and convert them to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundary.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinite distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def convert_junc_predictions(self, predictions, grid_size,
                                 detect_thresh=1 / 65, topk=300):
        """ Convert torch predictions to numpy arrays for evaluation. """
        # Convert to probability outputs first
        junc_prob = self.softmax(predictions, dim=1)
        junc_pred = junc_prob[:, :-1, :, :]

        junc_prob_np = np.transpose(junc_prob, axes=[0, 2, 3, 1])[:, :, :, :-1]
        junc_prob_np = np.sum(junc_prob_np, axis=-1)

        junc_pred_np = self.pixel_shuffle(junc_pred, grid_size)
        junc_pred_np = np.transpose(junc_pred_np, [0, 2, 3, 1])

        junc_pred_np_nms = self.super_nms(junc_pred_np, grid_size, detect_thresh, topk)
        # junc_pred_np = junc_pred_np.squeeze(-1)
        return {"junc_pred": junc_pred_np, "junc_pred_nms": junc_pred_np_nms,
                "junc_prob": junc_prob_np}

    def line_map_to_segments(self, junctions, line_map):
        """ Convert a line map to a Nx2x2 list of segments. """
        line_map_tmp = line_map.copy()

        output_segments = np.zeros([0, 2, 2])
        for idx in range(junctions.shape[0]):
            # if no connectivity, just skip it
            if line_map_tmp[idx, :].sum() == 0:
                continue
            # Record the line segment
            else:
                for idx2 in np.where(line_map_tmp[idx, :] == 1)[0]:
                    p1 = junctions[idx, :]  # HW format
                    p2 = junctions[idx2, :]
                    single_seg = np.concatenate([p1[None, ...], p2[None, ...]],
                                                axis=0)
                    output_segments = np.concatenate(
                        (output_segments, single_seg[None, ...]), axis=0)

                    # Update line_map
                    line_map_tmp[idx, idx2] = 0
                    line_map_tmp[idx2, idx] = 0

        return output_segments

    def postprocess(self, net_outputs):
        """
        return {
                'junctions': [],
                'heatmap': [],
                'line_segments': [],
                'descriptor', []
        }
        """
        junc_np = self.convert_junc_predictions(
            net_outputs["junctions"], self.grid_size,
            self.junc_detect_thresh, self.max_num_junctions)

        # print('junc_pred: ', junc_np['junc_pred'].reshape(-1)[:10])
        # print('junc_pred_nms: ', junc_np['junc_pred_nms'].reshape(-1)[:10])
        # print('junc_prob: ', junc_np['junc_prob'].reshape(-1)[:10])

        junctions = np.where(junc_np["junc_pred_nms"].squeeze())
        junctions = np.concatenate([junctions[0][..., None],
                                    junctions[1][..., None]], axis=-1)

        if net_outputs["heatmap"].shape[1] == 2:
            # Convert to single channel directly from here
            heatmap = self.softmax(net_outputs["heatmap"], dim=1)[:, 1:, :, :]
            heatmap = np.transpose(heatmap, [0, 2, 3, 1])
        else:
            heatmap = self.sigmoid(net_outputs["heatmap"])
            heatmap = np.transpose(heatmap, [0, 2, 3, 1])
        heatmap = heatmap[0, :, :, 0]

        # print('junctions: ', junctions)
        # Run the line detector.
        line_map, junctions, heatmap = self.line_detector.detect(junctions, heatmap)
        net_outputs["heatmap"] = heatmap
        net_outputs["junctions"] = junctions

        # If it's a line map with multiple detect_thresh and inlier_thresh
        if len(line_map.shape) > 2:
            num_detect_thresh = line_map.shape[0]
            num_inlier_thresh = line_map.shape[1]
            line_segments = []
            for detect_idx in range(num_detect_thresh):
                line_segments_inlier = []
                for inlier_idx in range(num_inlier_thresh):
                    line_map_tmp = line_map[detect_idx, inlier_idx, :, :]
                    line_segments_tmp = self.line_map_to_segments(junctions, line_map_tmp)
                    line_segments_inlier.append(line_segments_tmp)
                line_segments.append(line_segments_inlier)
        else:
            line_segments = self.line_map_to_segments(junctions, line_map)

        net_outputs["line_segments"] = line_segments
        return net_outputs

    def postprocess_step2(self, ref_detection, src_size_hw=(128, 128)):
        ref_line_seg = ref_detection["line_segments"]  # [N,2,2]
        ref_junctions = ref_detection['junctions']  # [N,2]
        # Round and convert junctions to int (and check the boundary)
        H, W = src_size_hw
        _scale = np.array([H/self.INPUT_H, W/self.INPUT_W], dtype=np.float64)
        ref_junctions = ref_junctions * _scale
        ref_line_seg = ref_line_seg * _scale

        junctions = (np.round(ref_junctions)).astype(np.int32)
        junctions[junctions < 0] = 0
        junctions[junctions[:, 0] >= H, 0] = H - 1  # (first dim) max bounded by H-1
        junctions[junctions[:, 1] >= W, 1] = W - 1  # (second dim) max bounded by W-1
        # print(ref_line_seg.shape)
        # print(ref_junctions.shape)
        # ref_descriptors = ref_detection["descriptor"][0]  # [N,H,W]
        # print(ref_descriptors.shape)

        junctions = junctions[:, ::-1]  # [N,2]
        lines = ref_line_seg[..., ::-1]  # [N,2,2]

        print('junctions: ', junctions.shape)
        print('lines: ', lines.shape)
        return junctions, lines

    def detect(self, img, thres=0.7):
        src_size_hw = img.shape[:2]
        mat_in = self.preprocess(img)
        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)
        ex.input(self.input_names[0], mat_in)

        s = time.time()
        outs = []
        for node in self.OUTPUT_NODES:
            assert node in self.output_names, f'{node} not in {self.output_names}'
            ret, out = ex.extract(node)  # [n,k,k]
            out = np.asarray(out)
            out = out[None, ...]
            # print(out.shape)
            outs.append(out)
        print(f'cnn forward() elasped : {time.time() - s} s', )


        mat_in.release()
        outputs = {"junctions": outs[0], "heatmap": outs[1]}
        ## 网络直接输出是对的
        # a = outputs['junctions']
        # b = outputs['heatmap']
        # print('junction: ', a.reshape(-1)[:10])
        # print('heatmap: ', b.reshape(-1)[:10])

        result = self.postprocess(outputs)
        # print(result['junctions'].shape)
        # print(result['line_segments'].shape)

        result = self.postprocess_step2(result, src_size_hw)
        return result


if __name__ == "__main__":
    print('hello')

    x = cv2.imread('./1514.png')
    # x = np.random.randint(0, 255, [128, 128, 3], dtype=np.uint8)
    m = NCNN_LineDetectNet()

    s = time.time()
    junctions, lines = m.detect(x)
    print(f'detect() elasped : {time.time() - s} s', )

    for junc in junctions:
        x1, y1 = list(map(int, junc))
        cv2.circle(x, (x1, y1), 2, (0, 255, 0), 2)

    for line in lines:
        line = np.int0(line).reshape(-1)
        x1, y1, x2, y2 = line
        cv2.line(x, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imwrite('./1514-out.jpg', x)
