import random
from typing import TypedDict

import numpy as np


class DinetBin(TypedDict):
    bin_id: int
    f_start: int
    f_end: int
    bbox: list[tuple[int, int, int, int]]


def compute_crop_radius(
    video_size, landmark_data_clip, random_scale=None, crop_type="square"
):
    """
    judge if crop face and compute crop radius
    """
    video_w, video_h = video_size[0], video_size[1]
    landmark_max_clip = np.max(landmark_data_clip, axis=1)
    if random_scale is None:
        random_scale = random.random() / 10 + 1.05
    else:
        random_scale = random_scale
    radius_h = (landmark_max_clip[:, 1] - landmark_data_clip[:, 29, 1]) * random_scale
    # nose -> chin
    radius_w = (
        landmark_data_clip[:, 54, 0] - landmark_data_clip[:, 48, 0]
    ) * random_scale
    # lip
    radius_clip = np.max(np.stack([radius_h, radius_w], 1), 1) // 2
    # max
    radius_max = np.max(radius_clip)
    radius_max = (np.int32(radius_max / 4) + 1) * 4
    radius_max_1_4 = radius_max // 4
    if crop_type == "dinet":
        clip_min_h = landmark_data_clip[:, 29, 1] - radius_max
        clip_max_h = landmark_data_clip[:, 29, 1] + radius_max * 2 + radius_max_1_4
        clip_min_w = landmark_data_clip[:, 33, 0] - radius_max - radius_max_1_4
        clip_max_w = landmark_data_clip[:, 33, 0] + radius_max + radius_max_1_4
    elif crop_type == "square-320":
        clip_min_h = landmark_data_clip[:, 29, 1] - radius_max * 2
        clip_max_h = landmark_data_clip[:, 29, 1] + radius_max * 2 + radius_max_1_4 * 4
        clip_min_w = landmark_data_clip[:, 33, 0] - radius_max * 2 - radius_max_1_4 * 2
        clip_max_w = landmark_data_clip[:, 33, 0] + radius_max * 2 + radius_max_1_4 * 2
    elif crop_type == "square":
        clip_min_h = landmark_data_clip[:, 29, 1] - radius_max - 2 * radius_max_1_4
        clip_max_h = landmark_data_clip[:, 29, 1] + radius_max * 3 - 2 * radius_max_1_4
        clip_min_w = landmark_data_clip[:, 33, 0] - radius_max * 2
        clip_max_w = landmark_data_clip[:, 33, 0] + radius_max * 2
    else:
        raise ValueError(f"Invalid crop type: {crop_type}")
    if min(clip_min_h.tolist() + clip_min_w.tolist()) < 0:
        return False, radius_max
    elif max(clip_max_h.tolist()) > video_h:
        return False, radius_max
    elif max(clip_max_w.tolist()) > video_w:
        return False, radius_max
    elif max(radius_clip) > min(radius_clip) * 1.5:
        return False, radius_max
    else:
        return True, radius_max


def dinet_bins(
    lms: list[list[tuple[int, int]] | None],
    height: int,
    width: int,
    clip_length: int = 9,
    crop_type: str = "square",
) -> tuple[list[DinetBin], list[DinetBin], int]:
    accepted_bins = []
    rejected_bins = []
    frame_length = len(lms)
    end_frame_index = list(range(clip_length, frame_length, clip_length))
    video_clip_num = len(end_frame_index)
    for bin_idx in range(video_clip_num):
        clip_lms = lms[
            end_frame_index[bin_idx] - clip_length : end_frame_index[bin_idx]
        ]
        if any([lm is None for lm in clip_lms]):
            continue
        crop_flag, radius_clip = compute_crop_radius(
            (width, height), np.array(clip_lms), crop_type=crop_type
        )
        # if not crop_flag:
        #     continue
        radius_clip_1_4 = radius_clip // 4
        bboxes = []
        for frame_index in range(
            end_frame_index[bin_idx] - clip_length, end_frame_index[bin_idx]
        ):
            frame_landmark = lms[frame_index]
            if crop_type == "square":
                y1 = frame_landmark[29][1] - radius_clip - radius_clip_1_4 * 2
                y2 = frame_landmark[29][1] + 3 * radius_clip - 2 * radius_clip_1_4
                x1 = frame_landmark[33][0] - 2 * radius_clip
                x2 = frame_landmark[33][0] + 2 * radius_clip
            elif crop_type == "dinet":
                y1 = frame_landmark[29][1] - radius_clip
                y2 = frame_landmark[29][1] + 2 * radius_clip + radius_clip_1_4
                x1 = frame_landmark[33][0] - radius_clip - radius_clip_1_4
                x2 = frame_landmark[33][0] + radius_clip + radius_clip_1_4
            elif crop_type == "square-320":
                y1 = frame_landmark[29][1] - radius_clip * 2
                y2 = frame_landmark[29][1] + radius_clip * 2 + 4 * radius_clip_1_4
                x1 = frame_landmark[33][0] - radius_clip * 2 - 2 * radius_clip_1_4
                x2 = frame_landmark[33][0] + radius_clip * 2 + 2 * radius_clip_1_4
            else:
                raise NotImplementedError(
                    "only dinet and square implemented. Please check code"
                )
            bboxes.append((round(x1), round(y1), round(x2), round(y2)))
        assert len(bboxes) == clip_length
        dinet_bin = {
            "bin_id": bin_idx,
            "f_start": end_frame_index[bin_idx] - clip_length,
            "f_end": end_frame_index[bin_idx],
            "bbox": bboxes,
        }

        if crop_flag:
            accepted_bins.append(dinet_bin)
        else:
            rejected_bins.append(dinet_bin)
    return accepted_bins, rejected_bins, video_clip_num
