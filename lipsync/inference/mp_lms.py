from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger
from tqdm import tqdm

MPLandmark = Any


def interpolate_landmarks(landmarks: np.ndarray, max_frames: int = 25, debug: bool = False) -> np.ndarray:
    """Interpolate missing landmarks (NaN values) in the sequence."""
    # Get indices where we have valid landmarks (not all NaN)
    valid_indices = ~np.isnan(landmarks[:, 0, 0])
    if debug:
        before_interpolation = valid_indices.sum()
    valid_frame_indices = np.where(valid_indices)[0]

    if len(valid_frame_indices) < 2:
        return landmarks

    interpolated = landmarks.copy()

    for i in range(len(valid_frame_indices) - 1):
        start, end = valid_frame_indices[i], valid_frame_indices[i + 1]
        gap = end - start

        if gap > 1 and gap <= max_frames:
            # Create interpolation for all coordinates at once
            start_lm = landmarks[start]
            end_lm = landmarks[end]

            # Generate intermediate frames
            for j in range(1, gap):
                alpha = j / gap
                interpolated[start + j] = start_lm * (1 - alpha) + end_lm * alpha

    if debug:
        valid_indices = ~np.isnan(interpolated[:, 0, 0])
        after_interpolation = valid_indices.sum()
        if after_interpolation != before_interpolation:
            logger.debug(
                f"Interpolated {before_interpolation} to {after_interpolation} landmarks, {~valid_indices.sum()} still invalid"
            )
    return interpolated


def get_video_landmarks(
    video_frames: list[np.ndarray],
    bboxes: np.ndarray | None = None,
    debug: bool = False,
) -> np.ndarray:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    # Initialize array with NaN values
    # MediaPipe face mesh has 478 landmarks with 3 coordinates (x,y,z)
    all_landmarks = np.full((len(video_frames), 478, 3), np.nan)
    if bboxes is not None:
        iterator = tqdm(enumerate(zip(video_frames, bboxes)), desc="Fetching landmarks", total=len(video_frames))
    else:
        iterator = tqdm(enumerate(video_frames), desc="Fetching landmarks", total=len(video_frames))

    for n, frame_bbox in iterator:
        if bboxes is not None:
            frame, bbox = frame_bbox
            x1, y1, x2, y2 = map(int, bbox)
            cropped_frame = frame[y1:y2, x1:x2]
            height, width = cropped_frame.shape[:2]
        else:
            cropped_frame = frame_bbox
            x1 = y1 = 0
            height, width = cropped_frame.shape[:2]

        if cropped_frame.size == 0:
            continue

        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            for i, lm in enumerate(landmarks):
                # Scale coordinates to the cropped frame size and then add the offset
                x = (lm.x * width) + x1
                y = (lm.y * height) + y1
                z = lm.z
                all_landmarks[n, i] = [x, y, z]

    face_mesh.close()
    # Interpolate missing landmarks
    all_landmarks = interpolate_landmarks(all_landmarks, debug=debug)

    return all_landmarks
