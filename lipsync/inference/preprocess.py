from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
import typer
from loguru import logger
from safetensors.torch import save_file
from tqdm import tqdm

from lipsync.create_ds.face_mask import get_face_mask, get_union_mask, make_square_bbox, mask2bbox
from lipsync.hallo_audio.processor import AudioProcessor
from lipsync.inference.mp_lms import get_video_landmarks


class CropType(str, Enum):
    SQUARE = "square"
    DINET = "dinet"
    SQUARE_320 = "square-320"
    WIDE = "wide"


def get_crop_size(crop_type: CropType, mouth_region_size: int) -> tuple[int, int]:
    radius = mouth_region_size // 2
    radius_1_4 = radius // 4
    if crop_type == CropType.SQUARE:
        return radius * 4, radius * 4
    elif crop_type == CropType.DINET:
        return radius * 3 + radius_1_4, radius * 2 + radius_1_4 * 2
    elif crop_type == CropType.SQUARE_320:
        return radius * 4 + 4 * radius_1_4, radius * 4 + 4 * radius_1_4
    elif crop_type == CropType.WIDE:
        return mouth_region_size * 2, mouth_region_size * 2
    else:
        raise ValueError(f"Invalid crop type: {crop_type}")


def save_cropped_frames(
    video_path: Path,
    dst_video: Path,
    mouth_region_size: int,
    crop_type: CropType,
    force: bool = False,
    debug: bool = False,
) -> str | None:
    try:
        dst_folder = dst_video / video_path.stem / "crops"
        dst_folder.mkdir(parents=True, exist_ok=True)
        existing_crops = list(dst_folder.glob("*.png"))
        bboxes_path = dst_video / video_path.stem / "bboxes.npy"
        video_meta = get_video_metadata(video_path)
        if not force and bboxes_path.exists():
            logger.debug(f"Skipping extracting frames for {video_path.stem} because it already exists")
            return None
        img_h, img_w = get_crop_size(crop_type, mouth_region_size)
        frames = get_video_frames(video_path)
        lms = get_video_landmarks(frames)
        if np.isnan(lms).any():
            raise ValueError("Mediapipe could not detect any faces in some frames of the video")
        lms_padded = np.pad(lms, ((2, 2), (0, 0), (0, 0)), mode="edge")
        bboxes = get_frame_bboxes(len(frames), lms_padded, video_meta["height"], video_meta["width"])
        np.save(dst_video / video_path.stem / "bboxes.npy", bboxes)
        if not force and len(existing_crops) == video_meta["num_frames"]:
            logger.debug(f"Skipping extracting frames for {video_path.stem} because it already exists")
            return None

        for i, bbox in tqdm(enumerate(bboxes), desc="Generating cropped frames", total=len(bboxes)):
            x1, y1, x2, y2, valid = bbox
            if valid != 1:
                raise ValueError("Some frames were rejected by wider crop criteria")
            frame = frames[i]

            # Get frame dimensions
            height, width = frame.shape[:2]

            # Clamp bounding box coordinates
            x1 = max(0, min(width, int(x1)))
            x2 = max(0, min(width, int(x2)))
            y1 = max(0, min(height, int(y1)))
            y2 = max(0, min(height, int(y2)))

            # Extract and process the face region
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (img_w, img_h))

            # Save the frame as PNG in the destination folder
            crops_path = dst_folder / f"{i:06d}.png"
            cv2.imwrite(str(crops_path), face)

        logger.debug(f"Successfully processed frames for video: {video_path.stem}")
        return None  # No error
    except Exception as e:
        if debug:
            logger.exception(e)
        return str(e)


def get_frame_bboxes(num_frames: int, landmarks: np.ndarray, height: int, width: int, expand_ratio: float = 2.3):
    bboxes = []
    for i in tqdm(range(num_frames)):
        _landmarks = landmarks[i : i + 5]
        # print(frame_numbers_req[i], _landmarks.shape)
        face_masks = []
        for _l in _landmarks:
            face_mask, _ = get_face_mask(_l, height, width, expand_ratio=expand_ratio)
            face_masks.append(face_mask)

        face_mask = get_union_mask(face_masks)
        face_valid, bbox = make_square_bbox(mask2bbox(face_mask), height, width)
        if not face_valid:
            bboxes.append([0, 0, 0, 0] + [-1])
            continue
        bboxes.append(bbox + [1])

    return np.asarray(bboxes)


def get_video_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Fetching frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        pbar.update(1)
    pbar.close()
    return frames


def get_video_metadata(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    return {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def save_audio_embedding(
    video_path: Path,
    audio_path: Path,
    dst_audio: Path,
    force: bool = False,
    device: str | None = None,
    debug: bool = False,
) -> str | None:
    try:
        dst_audio.parent.mkdir(parents=True, exist_ok=True)
        embed_file = dst_audio / video_path.stem / f"{audio_path.stem}.safetensors"
        if not force and embed_file.exists():
            logger.debug(f"Skipping extracting audio embedding for {audio_path.stem} because it already exists")
            return None  # Audio already exists

        video_meta = get_video_metadata(video_path)
        audio_embedding = get_audio_embedding(audio_path, video_meta["fps"], device=device)

        # Save as safetensors
        save_file({"audio_embedding": audio_embedding.contiguous()}, embed_file)

        logger.debug(f"Successfully processed and saved embedding: {audio_path.stem}")

        return None
    except Exception as e:
        if debug:
            logger.exception(e)
        return str(e)


def get_audio_embedding(
    video: Path,
    fps: float,
    device: str | None = None,
) -> torch.Tensor:
    audio_processor = AudioProcessor(16000, "facebook/wav2vec2-base-960h", False, device=device)
    audio = audio_processor.load_audio(str(video))
    logger.debug(f"Audio shape: {audio.shape}")
    audio_embedding, _ = audio_processor.preprocess(audio, fps)
    logger.debug(f"Audio embedding shape: {audio_embedding.shape}, device: {audio_embedding.device}")
    return audio_embedding


def run_preprocessing(
    video_path: Path,
    dst: Path,
    mouth_region_size: int,
    crop_type: CropType,
    audio_path: Path | None = None,
    force: bool = False,
    debug: bool = False,
):
    err = save_cropped_frames(video_path, dst, mouth_region_size, crop_type, force, debug)
    if err:
        return err
    if audio_path is not None:
        err = save_audio_embedding(video_path, audio_path, dst, force, debug=debug)
        if err:
            return err


if __name__ == "__main__":
    typer.run(run_preprocessing)
