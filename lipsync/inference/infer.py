import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import typer
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lipsync.dinet.model import DINetSPADE
from lipsync.inference.preprocess import (
    CropType,
    get_crop_size,
    get_video_frames,
    get_video_metadata,
    run_preprocessing,
)
from lipsync.inference.silero import get_silence_detector


def get_random_color(n=1):
    return np.random.randint(0, 255, (n, 3))


def print_once(msg):
    if not hasattr(print_once, "_printed_msgs"):
        print_once._printed_msgs = set()
    if msg not in print_once._printed_msgs:
        logger.info(msg)
        print_once._printed_msgs.add(msg)


def find_top_k_locations(array_2d, k=5):
    # Flatten the indices into 1D and get top k indices
    flat_indices = np.argpartition(array_2d.ravel(), -k)[-k:]

    # Sort these indices by their values to get them in descending order
    flat_indices = flat_indices[np.argsort(-array_2d.ravel()[flat_indices])]

    # Convert flat indices back to 2D coordinates
    row_indices = flat_indices // array_2d.shape[1]
    col_indices = flat_indices % array_2d.shape[1]

    return list(zip(row_indices, col_indices))


def get_model(ckpt_path: Path) -> tuple[DINetSPADE, str, str]:
    source_channel = 3
    ref_channel = 15
    upscale = 1
    seg_face = True
    use_attention = False
    model = DINetSPADE(source_channel, ref_channel, upscale=upscale, seg_face=seg_face, use_attention=use_attention)
    ckpt = torch.load(ckpt_path, weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    else:
        ckpt = ckpt
    ckpt_netg = {k[6:]: v for k, v in ckpt.items() if k.startswith("net_g.")}

    model.load_state_dict(ckpt_netg, strict=False)
    model.eval()
    epoch = ckpt["epoch"] if "epoch" in ckpt else "prod"
    dirpath = list(ckpt["callbacks"].items())[0][1]["dirpath"].split("/")[-2] if "callbacks" in ckpt else "prod"
    return model, epoch, dirpath


class InferenceDS(Dataset):
    def __init__(
        self,
        cache_dir: Path,  # source video
        ref_cache_dir: Path,  # this should be close mouth video
        audio_stem: str,
        lip_ref_cache_dir: Path | None = None,  # this should be open mouth video
        debug: bool = False,
        silence_regions: list[tuple[float, float]] = None,
        hide_frames: int = 0,
        put_ypr: bool = False,
        input_size: int = 256,
    ):
        self.cache_dir = cache_dir
        self.ref_cache_dir = ref_cache_dir
        self.lip_ref_cache_dir = lip_ref_cache_dir
        self.audio_stem = audio_stem
        self.debug = debug
        self.input_size = input_size
        self.put_ypr = put_ypr
        self.random_refs = None
        self.shut_mouth_img_tensor = None
        with torch.device("cpu"):
            self.audio_embed = load_file(self.cache_dir / f"{audio_stem}.safetensors")["audio_embedding"]
        self._get_landmarks()
        self.lms = np.concatenate([self.lms, self.lms[::-1]], axis=0)
        self._get_ypr_angles()
        self.frame_no = 0
        self.color = get_random_color(5)
        self.n_hide_frames = hide_frames
        self.change_frame_no = random.sample(range(5), self.n_hide_frames)
        # self._get_yaw_angles()
        # self._get_pitch_angles()
        self._get_mouth_openness()
        self.audio_embed = np.pad(self.audio_embed, ((2, 2), (0, 0), (0, 0)), mode="edge")
        images = list(self.cache_dir.glob("crops/*.png"))
        self.images = sorted(images, key=lambda x: int(x.stem))
        self.images += self.images[::-1]
        self.img_idxes = np.concatenate([np.arange(len(images)), np.arange(len(images))[::-1]])
        ref_images = list(self.ref_cache_dir.glob("crops/*.png"))
        self.ref_images = sorted(ref_images, key=lambda x: int(x.stem))
        self.ref_images += self.ref_images[::-1]
        self.ref_img_idxes = np.concatenate([np.arange(len(self.ref_images)), np.arange(len(self.ref_images))[::-1]])
        if lip_ref_cache_dir is not None:
            lip_ref_images = list(lip_ref_cache_dir.glob("crops/*.png"))
            self.lip_ref_images = sorted(lip_ref_images, key=lambda x: int(x.stem))
            self._get_lip_landmarks()
            self._get_lip_mouth_openness()
            self._get_lip_ypr_angles()
            self.lip_ref_img_idxes = np.arange(len(self.lip_ref_images))

        else:
            self.lip_ref_images = self.images
            self.lip_ref_img_idxes = self.img_idxes
            self.lip_openness = self.openness
            self.lip_lms = self.lms
            self.lip_yaw_angles = self.yaw_angles
            self.lip_pitch_angles = self.pitch_angles
            self.lip_roll_angles = self.roll_angles
        self._precompute_reference_frames()
        self.audio_window = 5
        self._refs = None
        self.silence_regions = silence_regions

    def _get_shut_mouth_img_tensor(self):
        if self.shut_mouth_img_tensor is None:
            shut_mouth_img_idx = np.argmin(self.openness)
            logger.info(f"shut_mouth_img_idx: {shut_mouth_img_idx}")
            shut_mouth_img = cv2.imread(str(self.images[shut_mouth_img_idx]))
            shut_mouth_img = cv2.resize(shut_mouth_img, (self.input_size, self.input_size))
            shut_mouth_img = shut_mouth_img.transpose(2, 0, 1)
            shut_mouth_img = shut_mouth_img / 255.0
            shut_mouth_img = torch.from_numpy(shut_mouth_img).unsqueeze(0)
            self.shut_mouth_img_tensor = shut_mouth_img
        return self.shut_mouth_img_tensor

    def _is_silence(self, img_idx: int) -> bool:
        for start, end in self.silence_regions:
            if start * 25 <= img_idx < end * 25:
                return True
        return False

    def _get_landmarks(self):
        lms = np.load(self.cache_dir / "lms.npy")
        bboxes = np.load(self.cache_dir / "bboxes.npy")

        def realign_landmarks(lm, x1, y1, x2, y2):
            lm[:, 0] = (lm[:, 0] - x1) / (x2 - x1) * 512
            lm[:, 1] = (lm[:, 1] - y1) / (y2 - y1) * 512
            return lm

        new_lms = []
        assert len(lms) == len(bboxes)
        for lm, bbox in zip(lms, bboxes):
            x1, y1, x2, y2, _ = bbox
            lm = realign_landmarks(lm, x1, y1, x2, y2)
            new_lms.append(lm)
        self.lms = np.stack(new_lms)

    def _get_lip_landmarks(self):
        def realign_landmarks(lm, x1, y1, x2, y2):
            lm[:, 0] = (lm[:, 0] - x1) / (x2 - x1) * 512
            lm[:, 1] = (lm[:, 1] - y1) / (y2 - y1) * 512
            return lm

        lms = np.load(self.lip_ref_cache_dir / "lms.npy")
        bboxes = np.load(self.lip_ref_cache_dir / "bboxes.npy")
        new_lms = []
        assert len(lms) == len(bboxes)
        for lm, bbox in zip(lms, bboxes):
            x1, y1, x2, y2, _ = bbox
            lm = realign_landmarks(lm, x1, y1, x2, y2)
            new_lms.append(lm)
        self.lip_lms = np.stack(new_lms)

    def _get_lip_mouth_openness(self):
        up = 13
        down = 14
        self.lip_openness = np.linalg.norm(self.lip_lms[:, up, :] - self.lip_lms[:, down, :], axis=1)

    def _get_lip_ypr_angles(self):
        lms = self.lip_lms
        NOSE_TIP = 4
        LEFT_EYE = 33
        RIGHT_EYE = 263
        nose = lms[:, NOSE_TIP]  # (batch_size, 3)
        left_eye = lms[:, LEFT_EYE]  # (batch_size, 3)
        right_eye = lms[:, RIGHT_EYE]  # (batch_size, 3)
        z_axis = np.zeros_like(nose)
        z_axis[:, 2] = 1.0  # Unit vector pointing into image
        x_axis = right_eye - left_eye
        x_axis = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis, axis=1, keepdims=True)

        # Recompute Z to ensure orthogonality
        z_axis = np.cross(x_axis, y_axis)

        # Stack to create rotation matrices: shape (batch_size, 3, 3)
        # Each column represents the world axis in face coordinates
        rotation_matrices = np.stack([x_axis, y_axis, z_axis], axis=2)

        # Extract Euler angles for XYZ rotation order
        # Based on https://www.geometrictools.com/Documentation/EulerAngles.pdf
        sy = np.sqrt(
            rotation_matrices[:, 0, 0] * rotation_matrices[:, 0, 0]
            + rotation_matrices[:, 1, 0] * rotation_matrices[:, 1, 0]
        )

        singular = sy < 1e-6

        # Handle regular case
        roll = np.arctan2(-rotation_matrices[:, 1, 2], rotation_matrices[:, 2, 2])
        pitch = np.arctan2(rotation_matrices[:, 0, 2], sy)
        yaw = np.arctan2(-rotation_matrices[:, 0, 1], rotation_matrices[:, 0, 0])

        # Handle gimbal lock case
        roll[singular] = np.arctan2(rotation_matrices[singular, 2, 1], rotation_matrices[singular, 1, 1])
        pitch[singular] = np.arctan2(rotation_matrices[singular, 0, 2], sy[singular])
        yaw[singular] = 0
        self.lip_yaw_angles = yaw
        self.lip_pitch_angles = pitch
        self.lip_roll_angles = roll

    def _get_yaw_angles(self):
        lms = self.lms
        left = 234
        right = 454
        nose = 5
        self.yaw_angles = np.log(
            np.linalg.norm(lms[:, left, :] - lms[:, nose, :], axis=1)
            / np.linalg.norm(lms[:, right, :] - lms[:, nose, :], axis=1)
            + 1e-6
        )

    def _get_pitch_angles(self):
        lms = self.lms
        top = 10
        nose = 4
        bottom = 152
        self.pitch_angles = np.log(
            np.linalg.norm(lms[:, top, :] - lms[:, nose, :], axis=1)
            / np.linalg.norm(lms[:, bottom, :] - lms[:, nose, :], axis=1)
            + 1e-6
        )

    def _get_ypr_angles(self):
        lms = self.lms
        NOSE_TIP = 4
        LEFT_EYE = 33
        RIGHT_EYE = 263
        nose = lms[:, NOSE_TIP]  # (batch_size, 3)
        left_eye = lms[:, LEFT_EYE]  # (batch_size, 3)
        right_eye = lms[:, RIGHT_EYE]  # (batch_size, 3)
        z_axis = np.zeros_like(nose)
        z_axis[:, 2] = 1.0  # Unit vector pointing into image
        x_axis = right_eye - left_eye
        x_axis = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis, axis=1, keepdims=True)

        # Recompute Z to ensure orthogonality
        z_axis = np.cross(x_axis, y_axis)

        # Stack to create rotation matrices: shape (batch_size, 3, 3)
        # Each column represents the world axis in face coordinates
        rotation_matrices = np.stack([x_axis, y_axis, z_axis], axis=2)

        # Extract Euler angles for XYZ rotation order
        # Based on https://www.geometrictools.com/Documentation/EulerAngles.pdf
        sy = np.sqrt(
            rotation_matrices[:, 0, 0] * rotation_matrices[:, 0, 0]
            + rotation_matrices[:, 1, 0] * rotation_matrices[:, 1, 0]
        )

        singular = sy < 1e-6

        # Handle regular case
        roll = np.arctan2(-rotation_matrices[:, 1, 2], rotation_matrices[:, 2, 2])
        pitch = np.arctan2(rotation_matrices[:, 0, 2], sy)
        yaw = np.arctan2(-rotation_matrices[:, 0, 1], rotation_matrices[:, 0, 0])

        # Handle gimbal lock case
        roll[singular] = np.arctan2(rotation_matrices[singular, 2, 1], rotation_matrices[singular, 1, 1])
        pitch[singular] = np.arctan2(rotation_matrices[singular, 0, 2], sy[singular])
        yaw[singular] = 0
        self.yaw_angles = yaw
        self.pitch_angles = pitch
        self.roll_angles = roll

    def _get_mouth_openness(self):
        up = 13
        down = 14
        self.openness = np.linalg.norm(self.lms[:, up, :] - self.lms[:, down, :], axis=1)

    def __len__(self):
        return len(self.audio_embed) - 4

    def _precompute_reference_frames(
        self, num_yaw_buckets: int = 5, num_pitch_buckets: int = 3
    ):  # -> dict[int, list[int]]:
        """Precompute reference frames for each frame."""
        logger.info("Precomputing reference frames")
        logger.info(f"lip_ref_images exists: {self.lip_ref_images is not None}")
        if self.lip_ref_images is not None:
            # lip refs are top 4 frames with highest lip openness
            lip_openness = self.lip_openness
            lip_refs = np.argsort(lip_openness).tolist()[-4:]
            logger.info(f"lip_refs: {lip_refs} out of {len(lip_openness)}")
            self.reference_indices = {i: [i] + lip_refs for i in range(len(self.images))}
        else:
            yaw_boundaries = []
            min_yaw, max_yaw = self.yaw_angles.min(), self.yaw_angles.max()
            for i in range(num_yaw_buckets):
                yaw_boundaries.append(min_yaw + (max_yaw - min_yaw) * i / num_yaw_buckets)
            yaw_boundaries.append(max_yaw)
            yaw_boundaries = np.unique(yaw_boundaries)
            pitch_boundaries = []
            min_pitch, max_pitch = self.pitch_angles.min(), self.pitch_angles.max()
            for i in range(num_pitch_buckets):
                pitch_boundaries.append(min_pitch + (max_pitch - min_pitch) * i / num_pitch_buckets)
            pitch_boundaries.append(max_pitch)
            pitch_boundaries = np.unique(pitch_boundaries)
            yaw_buckets = []
            pitch_buckets = []
            for yaw_angle in self.yaw_angles:
                for idx, yaw_boundary in enumerate(list(yaw_boundaries)):
                    if yaw_angle < yaw_boundary:
                        break
                yaw_buckets.append(idx - 1)
            for pitch_angle in self.pitch_angles:
                for idx, pitch_boundary in enumerate(list(pitch_boundaries)):
                    if pitch_angle < pitch_boundary:
                        break
                pitch_buckets.append(idx - 1)
            self.yaw_buckets = yaw_buckets
            self.pitch_buckets = pitch_buckets
            index_with_open_mouth_per_bucket = [[-1] * num_pitch_buckets for _ in range(num_yaw_buckets)]
            num_frames_per_bucket = [[0] * num_pitch_buckets for _ in range(num_yaw_buckets)]
            for i in range(num_yaw_buckets):
                for j in range(num_pitch_buckets):
                    tmp = self.openness * (np.array(yaw_buckets) == i) * (np.array(pitch_buckets) == j)
                    if tmp.max() > 0:
                        index_with_open_mouth_per_bucket[i][j] = np.argmax(tmp)
                    else:
                        index_with_open_mouth_per_bucket[i][j] = -1
                    num_frames_per_bucket[i][j] = ((np.array(yaw_buckets) == i) * (np.array(pitch_buckets) == j)).sum()
            print("num_frames_per_bucket:")
            for i in range(num_pitch_buckets):
                for j in range(num_yaw_buckets):
                    print(f"{num_frames_per_bucket[j][i]:3d}", end=" ")
                print()
            self.index_with_open_mouth_per_bucket = index_with_open_mouth_per_bucket

            # Get coordinates of top buckets with highest frame counts
            top_bucket_coords = find_top_k_locations(np.array(num_frames_per_bucket), 10)

            # Now find actual frame indices from these buckets
            open_mouth_indices = []
            for yaw_idx, pitch_idx in top_bucket_coords:
                # Find frames belonging to this (yaw, pitch) bucket
                bucket_frames = np.where((np.array(yaw_buckets) == yaw_idx) & (np.array(pitch_buckets) == pitch_idx))[0]

                if len(bucket_frames) > 0:
                    # If bucket has frames, get the one with highest openness
                    best_frame = bucket_frames[np.argmax(self.openness[bucket_frames])]
                    open_mouth_indices.append(int(best_frame))  # Convert to int to be safe

            openness = self.openness[open_mouth_indices]
            # find top 3 indices with highest openness
            top_3_indices = np.argsort(openness)[-100:]
            open_mouth_indices = [open_mouth_indices[i] for i in top_3_indices]

            print(f"open_mouth_indices: {open_mouth_indices}")
            print(f"Mouth openness: {self.openness[open_mouth_indices]}")
            print(f"top 3 open mouth locations: {top_3_indices}")

            self.reference_indices = {
                i: [
                    i,
                    index_with_open_mouth_per_bucket[self.yaw_buckets[i]][self.pitch_buckets[i]]
                    if index_with_open_mouth_per_bucket[self.yaw_buckets[i]][self.pitch_buckets[i]] != -1
                    else i,
                ]
                + random.sample(open_mouth_indices, 3)
                for i in range(len(self.images))
            }

    def get_refs(self, img_idx: int):
        """Get reference frames for the current image."""
        self.frame_no += 1
        if self.frame_no % 100 == 0:
            self.color = get_random_color(5)
            self.change_frame_no = random.sample(range(5), self.n_hide_frames)
        ref_indices = self.reference_indices[img_idx].copy()
        if self.random_refs is None:
            self.random_refs = [j for j in random.sample(range(len(self.images)), 4)]
        ref_indices = self.random_refs
        # ref_indices = [int((img_idx // 25) * 25)] + self.random_refs
        ref_indices = [img_idx] + ref_indices
        if len(ref_indices) == 4:
            # ref_indices = [int((img_idx // 25) * 25)] + ref_indices
            ref_indices = [img_idx] + ref_indices
        elif len(ref_indices) != 5:
            raise ValueError(f"Invalid number of reference frames: {len(ref_indices)}")
        refs = []
        # is_silence = self._is_silence(img_idx) or True
        for i, idx in enumerate(ref_indices):
            if i == 0:
                ref_image = cv2.imread(str(self.ref_images[idx]))
            else:
                if self.lip_ref_images is not None:
                    ref_image = cv2.imread(str(self.lip_ref_images[idx]))
                else:
                    ref_image = cv2.imread(str(self.images[idx]))
            ref_image = cv2.resize(ref_image, (self.input_size, self.input_size))
            if i in self.change_frame_no:
                ref_image = ref_image * 0 + self.color[i]
            # write yaw angle to image, top middle
            elif self.put_ypr:
                cv2.putText(
                    ref_image,
                    f"YAW: {self.yaw_angles[idx]:.2f}",
                    (0, 512 // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    ref_image,
                    f"PITCH: {self.pitch_angles[idx]:.2f}",
                    (0, 512 // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    ref_image,
                    f"LD: {self.openness[idx]:.2f}",
                    (0, 512 // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
            if self.debug:
                ref_path = self.cache_dir / "refs" / f"frame_{img_idx}_ref_{idx}.png"
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ref_path), ref_image)
            ref_array = ref_image / 255.0
            refs.append(ref_array)

        refs = np.array(refs)
        refs = torch.from_numpy(np.concatenate(refs, axis=2)).permute(2, 0, 1).to(torch.float32)
        return refs  # torch.cat((ref_img, refs), 0)

    def __getitem__(self, idx: int):
        audio_embed = self.audio_embed[idx : idx + self.audio_window]
        img_idx = self.img_idxes[idx % len(self.images)]
        image = cv2.imread(str(self.images[img_idx]))
        image = cv2.resize(image, (self.input_size, self.input_size))
        # write yaw angle to image, top middle
        if self.put_ypr:
            cv2.putText(
                image,
                f"YAW: {self.yaw_angles[img_idx]:.2f}",
                (0, self.input_size // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                image,
                f"PITCH: {self.pitch_angles[img_idx]:.2f}",
                (0, self.input_size // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            # lip distance
            cv2.putText(
                image,
                f"LD: {self.openness[img_idx]:.2f}",
                (0, self.input_size // 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
        image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32)
        audio_embed = torch.from_numpy(audio_embed).to(torch.float32)
        return image, self.get_refs(img_idx), audio_embed, img_idx


def add_audio_to_video(face_video_path: Path, audio_path: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        # log only error
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(face_video_path),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(dst),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e}")


def main(
    ckpt_path: Path = typer.Argument(..., help="Path to the dinet checkpoint"),
    video_path: Path = typer.Argument(..., help="Path to the video"),
    audio_path: Path = typer.Argument(..., help="Path to the audio"),
    dst: Path = typer.Argument(..., help="Path to the output directory"),
    reference_video_path: Path | None = typer.Option(None, help="Path to first reference frames video"),
    lip_reference_video_path: Path | None = typer.Option(None, help="Path to last 4 reference frames video"),
    suffix: str = typer.Option("", help="Suffix to the output video name"),
    input_size: int = typer.Option(256, help="Input size"),
    mouth_region_size: int = typer.Option(128, help="Mouth region size"),
    crop_type: CropType = typer.Option(CropType.WIDE, help="Crop type"),
    force: bool = typer.Option(False, help="Force overwrite"),
    debug: bool = typer.Option(False, help="Debug mode"),
    no_refs: bool = typer.Option(False, help="No reference frames at the bottom of the video"),
    plot_grid: bool = typer.Option(False, help="Plot grid on the output video"),
    device: int = typer.Option(0, help="Device to use"),
    hide_frames: int = typer.Option(0, help="Number of frames to hide"),
    hidden_features: bool = typer.Option(False, help="Get hidden features"),
    mask_first_ref_lip: bool = typer.Option(False, help="Mask first reference lip"),
    no_skin: bool = typer.Option(False, help="No skin source frame"),
    show_idexs: str = typer.Option(None, help="Comma separated indexes of segmentation to show"),
    auto_regressive: bool = typer.Option(False, help="Use Auto regressi mode"),
    auto_shut_mouth: bool = typer.Option(False, help="Use Auto shut mouth mode"),
):
    if show_idexs is not None:
        show_idexs = [int(i) for i in show_idexs.split(",")]
    try:
        err = run_preprocessing(
            video_path=video_path,
            dst=dst,
            mouth_region_size=mouth_region_size,
            crop_type=crop_type,
            audio_path=audio_path,
            force=force,
            debug=debug,
        )
        logger.info(f"Preprocessing error: {err}")
        if reference_video_path is not None:
            err = run_preprocessing(
                video_path=reference_video_path,
                dst=dst,
                mouth_region_size=mouth_region_size,
                crop_type=crop_type,
                audio_path=None,
                force=force,
                debug=debug,
            )
            logger.info(f"Preprocessing reference video error: {err}")
        if lip_reference_video_path is not None:
            err = run_preprocessing(
                video_path=lip_reference_video_path,
                dst=dst,
                mouth_region_size=mouth_region_size,
                crop_type=crop_type,
                audio_path=None,
                force=force,
                debug=debug,
            )
            logger.info(f"Preprocessing lip reference video error: {err}")
        if err:
            logger.error(err)
            return
        device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        if reference_video_path is None:
            reference_video_path = video_path
        model, epoch, run_name = get_model(ckpt_path)
        model = model.to(device)
        model.segface.model.to(device)
        silence_detector = get_silence_detector()
        silence_regions = silence_detector.detect_silence(audio_path)
        print(f"silence_regions: {silence_regions}")
        if plot_grid:
            suffix = f"{suffix}_with_grid"
        if hide_frames > 0:
            suffix = f"{suffix}_with_hide_{hide_frames}_frames"
        if not no_refs:
            suffix = f"{suffix}_with_refs"
        if auto_shut_mouth:
            suffix = f"{suffix}_with_auto_shut_mouth"
        ds = InferenceDS(
            dst / video_path.stem,
            dst / reference_video_path.stem,
            audio_path.stem,
            dst / lip_reference_video_path.stem if lip_reference_video_path is not None else None,
            debug,
            silence_regions,
            hide_frames,
            input_size=input_size,
        )
        shut_mouth_img_tensor = None
        if auto_shut_mouth:
            shut_mouth_img_tensor = ds._get_shut_mouth_img_tensor()
        mask_coords = [128, 224, 384, 480] if input_size == 512 else [64, 112, 192, 240]
        bs = 1
        num_workers = min(bs, 32)
        dl = DataLoader(ds, batch_size=bs, num_workers=num_workers, shuffle=False)
        metadata = get_video_metadata(video_path)
        img_h, img_w = get_crop_size(crop_type, mouth_region_size)
        out_face_video = dst / f"{video_path.stem}_{audio_path.stem}_face.mp4"
        out_face_video.parent.mkdir(parents=True, exist_ok=True)
        out_face = cv2.VideoWriter(
            str(out_face_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata["fps"],
            (img_w, img_h),
        )

        if suffix:
            suffix = f"RUN_{run_name}_EPOCH_{epoch}_{suffix}"
        else:
            suffix = f"RUN_{run_name}_EPOCH_{epoch}"
        if auto_regressive:
            suffix = f"{suffix}_auto_regressive"
        out_video = dst / f"{video_path.stem}_{audio_path.stem}_{suffix}.mp4"
        out_video.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata["fps"],
            (
                metadata["width"],
                metadata["height"] + (1 * (1 - no_refs)) * metadata["width"] // 5,
            ),  # additional row for refs
        )
        if hidden_features:
            hidden_features_pre_video_path = dst / f"{video_path.stem}_{audio_path.stem}_hidden_features_pre.mp4"
            hidden_features_post_video_path = dst / f"{video_path.stem}_{audio_path.stem}_hidden_features_post.mp4"
            hidden_features_pre_video_path.parent.mkdir(parents=True, exist_ok=True)
            hidden_features_post_video_path.parent.mkdir(parents=True, exist_ok=True)
            hidden_features_pre_video = cv2.VideoWriter(
                str(hidden_features_pre_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                metadata["fps"],
                (1024, 1024),
            )
            hidden_features_post_video = cv2.VideoWriter(
                str(hidden_features_post_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                metadata["fps"],
                (1024, 1024),
            )
        bboxes = np.load(dst / video_path.stem / "bboxes.npy")
        frames = get_video_frames(video_path)
        output_index = 0
        pred_AR = None
        for image, refs, audio_embed, img_idxes in tqdm(dl, desc="Generating predicted faces"):
            # with torch.no_grad():
            image = image.to(device)
            refs = refs.to(device)
            if shut_mouth_img_tensor is not None:
                shut_mouth_img_tensor = shut_mouth_img_tensor.to(device)
            if auto_regressive:
                print_once("Using Auto Regressive")
                if pred_AR is not None and output_index % 25 > 0:
                    print_once("Pred AR is not None")
                    refs[:, :3, ...] = pred_AR.detach().clone()
                else:
                    print_once("Pred AR is None")
            audio_embed = audio_embed.to(device)
            with torch.no_grad():
                if hidden_features:
                    pred, _, hidden_features_pre, hidden_features_post, refs = model(
                        image,
                        refs,
                        audio_embed,
                        mask_dim=mask_coords,
                        get_features=True,
                        mask_lip=mask_first_ref_lip,
                        no_skin=no_skin,
                        show_idexs=show_idexs,
                        shut_mouth=shut_mouth_img_tensor if auto_shut_mouth else None,
                    )
                else:
                    pred, _, __, ___, refs = model(
                        image,
                        refs,
                        audio_embed,
                        mask_dim=mask_coords,
                        mask_lip=mask_first_ref_lip,
                        no_skin=no_skin,
                        show_idexs=show_idexs,
                        shut_mouth=shut_mouth_img_tensor if auto_shut_mouth else None,
                    )
            pred_AR = pred.clone()
            print_once(f"pred_AR.shape: {pred_AR.shape}")
            pred = pred.permute(0, 2, 3, 1).cpu().numpy() * 255
            refs = rearrange(refs, "b (f c) h w -> b c h (f w)", f=5)
            ref_w = metadata["width"]
            ref_h = refs.shape[2] * ref_w // refs.shape[3]
            refs = F.interpolate(refs, size=(ref_h, ref_w), mode="bilinear")
            refs = refs.permute(0, 2, 3, 1).cpu().numpy() * 255
            refs = refs.astype(np.uint8)
            if hidden_features:
                hidden_features_pre = rearrange(hidden_features_pre, "b (f c) h w -> b 1 (f h) (c w)", f=16).repeat(
                    1, 3, 1, 1
                )
                hidden_features_post = rearrange(hidden_features_post, "b (f c) h w -> b 1 (f h) (c w)", f=16).repeat(
                    1, 3, 1, 1
                )
                # convert from -1 to 0 and 1 to 255
                min_clip, max_clip = -2.0, 8.0
                hidden_features_pre = (hidden_features_pre - min_clip) / (max_clip - min_clip)
                hidden_features_post = (hidden_features_post - min_clip) / (max_clip - min_clip)
                hidden_features_pre = hidden_features_pre.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255
                hidden_features_post = hidden_features_post.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255
                # hidden_features_pre = hidden_features_pre.permute(0, 2, 3, 1).cpu().numpy() * 255
                # hidden_features_post = hidden_features_post.permute(0, 2, 3, 1).cpu().numpy() * 255
                hidden_features_pre = hidden_features_pre.astype(np.uint8)
                hidden_features_post = hidden_features_post.astype(np.uint8)

                # print(f"hidden_features_pre.min(): {hidden_features_pre.min()}, hidden_features_pre.max(): {hidden_features_pre.max()}")
                # print(f"hidden_features_post.min(): {hidden_features_post.min()}, hidden_features_post.max(): {hidden_features_post.max()}")
            for i in range(pred.shape[0]):
                out_face.write(pred[i].astype(np.uint8))
                img_idx = img_idxes[i]
                frame = frames[img_idx].copy()
                bbox = bboxes[img_idx]
                x1, y1, x2, y2, _ = bbox
                orig_i = image[i].permute(1, 2, 0).cpu().numpy() * 255
                orig_i = orig_i.astype(np.uint8)
                # pred_i = pred[i][112:240, 64:192, :].astype(np.uint8)
                pred_i = pred[i].astype(np.uint8)  # [224:480, 128:384, :].astype(np.uint8)
                # draw rectange on pred_i
                # convert pred_i to cv2 image
                # pred_i = cv2.cvtColor(pred_i, cv2.COLOR_RGB2BGR)
                # cv2.rectangle(pred_i, (128, 224), (384, 480), (0, 0, 255), 2)
                # pred_i = cv2.cvtColor(pred_i, cv2.COLOR_BGR2RGB)
                # output_i = cv2.seamlessClone(
                #     pred_i, orig_i, np.ones(pred_i.shape, dtype=np.uint8) * 255, (128, 176), cv2.NORMAL_CLONE
                # )
                # output_i = pred[i].astype(np.uint8)
                pad_x1 = int(max(0, -x1))
                pad_y1 = int(max(0, -y1))
                pad_x2 = int(max(0, x2 - frame.shape[1]))
                pad_y2 = int(max(0, y2 - frame.shape[0]))
                this_pred = cv2.resize(pred_i, (int(x2 - x1), int(y2 - y1)))
                new_x1 = int(x1 + pad_x1)
                new_y1 = int(y1 + pad_y1)
                new_x2 = int(x2 + pad_x1 - pad_x2)
                new_y2 = int(y2 + pad_y1 - pad_y2)
                frame[new_y1:new_y2, new_x1:new_x2] = this_pred[
                    pad_y1 : pad_y1 + new_y2 - new_y1, pad_x1 : pad_x1 + new_x2 - new_x1
                ]
                if not no_refs:
                    frame = np.concatenate([frame, refs[i]], axis=0)
                pred_path = dst / video_path.stem / "preds" / f"frame_{output_index+i}.png"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(pred_path), frame)
                out.write(frame)
                if hidden_features:
                    hidden_features_pre_video.write(hidden_features_pre[i])
                    hidden_features_post_video.write(hidden_features_post[i])
            output_index += pred.shape[0]
        out_face.release()
        out.release()
        if hidden_features:
            hidden_features_pre_video.release()
            hidden_features_post_video.release()
        audio_video_path = dst / f"{out_face_video.stem}_audio.mp4"
        add_audio_to_video(out_face_video, audio_path, audio_video_path)
        out_face_video.unlink()
        audio_video_path.rename(out_face_video)
        audio_video_path = dst / f"{out_video.stem}_audio.mp4"
        add_audio_to_video(out_video, audio_path, audio_video_path)
        out_video.unlink()
        audio_video_path.rename(out_video)
        logger.info(f"Successfully generated {out_face_video.absolute()} and {out_video.absolute()}")
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    typer.run(main)

    # from einops import rearrange
    # from torchvision.utils import save_image

    # ds = InferenceDS(Path("temp/anshul3-30s"), "anshul-driving-audio-30s")
    # image, refs, audio_embed, img_idx = ds[1801]
    # image_path = Path("temp/anshul3-30s/test/image.png")
    # image_path.parent.mkdir(parents=True, exist_ok=True)
    # save_image(image, image_path)
    # ref_path = Path("temp/anshul3-30s/test/refs.png")
    # ref_path.parent.mkdir(parents=True, exist_ok=True)
    # refs = rearrange(refs, "(f c) h w -> f c h w", f=5)
    # save_image(refs, ref_path)
    # print(image.shape, refs.shape, audio_embed.shape, img_idx)
