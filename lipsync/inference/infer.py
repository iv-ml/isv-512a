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


def get_model(ckpt_path: Path) -> DINetSPADE:
    source_channel = 3
    ref_channel = 15
    upscale = 1
    seg_face = True
    use_attention = False
    model = DINetSPADE(source_channel, ref_channel, upscale=upscale, seg_face=seg_face, use_attention=use_attention)
    ckpt = torch.load(ckpt_path, weights_only=True)
    ckpt_netg = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net_g.")}

    model.load_state_dict(ckpt_netg)
    model.eval()
    return model


class InferenceDS(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        ref_cache_dir: Path,
        audio_stem: str,
        debug: bool = False,
        silence_regions: list[tuple[float, float]] = None,
    ):
        self.cache_dir = cache_dir
        self.ref_cache_dir = ref_cache_dir
        self.audio_stem = audio_stem
        self.debug = debug
        self.audio_embed = load_file(self.cache_dir / f"{audio_stem}.safetensors")["audio_embedding"]
        self.audio_embed = np.pad(self.audio_embed, ((2, 2), (0, 0), (0, 0)), mode="edge")
        images = list(self.cache_dir.glob("crops/*.png"))
        self.images = sorted(images, key=lambda x: int(x.stem))
        self.images += self.images[::-1]
        self.img_idxes = np.concatenate([np.arange(len(images)), np.arange(len(images))[::-1]])
        ref_images = list(self.ref_cache_dir.glob("crops/*.png"))
        self.ref_images = sorted(ref_images, key=lambda x: int(x.stem))
        self.ref_images += self.ref_images[::-1]
        self.ref_img_idxes = np.concatenate([np.arange(len(self.ref_images)), np.arange(len(self.ref_images))[::-1]])
        self.reference_indices = self._precompute_reference_frames()
        self.audio_window = 5
        self._refs = None
        self.silence_regions = silence_regions

    def _is_silence(self, idx: int) -> bool:
        if self.silence_regions is None:
            return False
        for start, end in self.silence_regions:
            if start * 25.0 <= idx < end * 25.0:
                return True
        return False

    def __len__(self):
        return len(self.audio_embed) - 4

    def _precompute_reference_frames(self) -> dict[int, list[int]]:
        """Precompute reference frames for each frame."""
        random_indices = np.random.randint(0, len(self.ref_images), size=4)
        random_indices = np.asarray([2705, 513, 193, 2713, 310])  # works best for anshuls video
        return {i: random_indices for i in range(len(self.images))}

    def get_refs(self, img_idx: int):
        """Get reference frames for the current image."""
        ref_indices = self.reference_indices[img_idx]
        if len(ref_indices) == 4:
            ref_indices = [img_idx] + [self.img_idxes[idx] for idx in ref_indices]
        elif len(ref_indices) == 5:
            ref_indices = [self.img_idxes[idx] for idx in ref_indices]
        else:
            raise ValueError(f"Invalid number of reference frames: {len(ref_indices)}")
        refs = []
        is_silence = self._is_silence(img_idx)
        for i, idx in enumerate(ref_indices):
            if is_silence and i == 0:
                ref_image = cv2.imread(str(self.images[idx]))
            else:
                ref_image = cv2.imread(str(self.ref_images[idx]))
            ref_image = cv2.resize(ref_image, (512, 512))
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
        image = cv2.resize(image, (512, 512))
        image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32)
        audio_embed = torch.from_numpy(audio_embed).to(torch.float32)
        return image, self.get_refs(img_idx), audio_embed, img_idx


def add_audio_to_video(face_video_path: Path, audio_path: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
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
    reference_video_path: Path | None = typer.Option(None, help="Path to the reference video"),
    suffix: str = typer.Option("", help="Suffix to the output video name"),
    mouth_region_size: int = typer.Option(128, help="Mouth region size"),
    crop_type: CropType = typer.Option(CropType.WIDE, help="Crop type"),
    force: bool = typer.Option(False, help="Force overwrite"),
    debug: bool = typer.Option(False, help="Debug mode"),
    add_stats: bool = typer.Option(False, help="Add stats to the output video"),
    plot_grid: bool = typer.Option(False, help="Plot grid on the output video"),
):
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
        if err:
            logger.error(err)
            return
        device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        if reference_video_path is None:
            reference_video_path = video_path
        model = get_model(ckpt_path).to(device)
        model.segface.model.to(device)
        silence_detector = get_silence_detector()
        silence_regions = silence_detector.detect_silence(audio_path)
        print(f"silence_regions: {silence_regions}")
        ds = InferenceDS(
            dst / video_path.stem, dst / reference_video_path.stem, audio_path.stem, debug, silence_regions
        )
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
            out_video = dst / f"{video_path.stem}_{audio_path.stem}_{suffix}.mp4"
        else:
            out_video = dst / f"{video_path.stem}_{audio_path.stem}.mp4"
        out_video.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata["fps"],
            (
                metadata["width"],
                metadata["height"] + (1 + 1 * add_stats) * metadata["width"] // 5,
            ),  # additional row for stats
        )
        bboxes = np.load(dst / video_path.stem / "bboxes.npy")
        frames = get_video_frames(video_path)
        for image, refs, audio_embed, img_idxes in tqdm(dl, desc="Generating predicted faces"):
            # with torch.no_grad():
            image = image.to(device)
            refs = refs.to(device)
            audio_embed = audio_embed.to(device)
            if not add_stats:
                with torch.no_grad():
                    pred, _ = model(image, refs, audio_embed, mask_dim=[128, 224, 384, 480])
            else:
                (
                    pred,
                    saliency,
                    image_saliency,
                    max_aud_saliency,
                    min_aud_saliency,
                    mean_aud_saliency,
                    lips_region,
                    input_image,
                ) = predict_ref_influencing_lips(model, image, refs, audio_embed, [128, 224, 384, 480], device)
                # draw grid lines on refs and saliency
                pred = pred.detach()
                refs = refs.detach()
                saliency = saliency.detach()
                image_saliency = image_saliency.detach()
                image = image.detach()
                lips_region = lips_region.detach()
                input_image = input_image.detach()
                if plot_grid:
                    for i in range(1, 8):
                        refs[:, :, i * 32, :] = 1
                        refs[:, :, :, i * 32] = 1
                        saliency[:, :, i * 32, :] = 1
                        saliency[:, :, :, i * 32] = 1
                        image_saliency[:, :, i * 32, :] = 1
                        image_saliency[:, :, :, i * 32] = 1
                        pred[:, :, i * 32 * 3, :] = 0
                        pred[:, :, :, i * 32 * 3] = 0
                        lips_region[:, :, i * 32, :] = 0
                        lips_region[:, :, :, i * 32] = 0
                        input_image[:, :, i * 32, :] = 0
                        input_image[:, :, :, i * 32] = 0
                    refs[:, :, 0, :] = 0
                    refs[:, :, -1, :] = 0
                    refs[:, :, :, 0] = 0
                    refs[:, :, :, -1] = 0
                    saliency[:, :, 0, :] = 0
                    saliency[:, :, :, 0] = 0
                    saliency[:, :, :, -1] = 0
                    saliency[:, :, :, -1] = 0
                    image_saliency[:, :, 0, :] = 0
                    image_saliency[:, :, :, 0] = 0
                    image_saliency[:, :, :, -1] = 0
                    image_saliency[:, :, :, -1] = 0
                    input_image[:, :, 0, :] = 0
                    input_image[:, :, :, 0] = 0
                    input_image[:, :, :, -1] = 0
                    input_image[:, :, :, -1] = 0
                    lips_region[:, :, 0, :] = 0
                    lips_region[:, :, :, 0] = 0
                    lips_region[:, :, :, -1] = 0
                    lips_region[:, :, :, -1] = 0
                lips_region = lips_region.permute(0, 2, 3, 1).cpu().numpy() * 255.0
                lips_region = lips_region.astype(np.uint8)
                input_image = input_image.permute(0, 2, 3, 1).cpu().numpy() * 255
                input_image = input_image.astype(np.uint8)
            pred = pred.permute(0, 2, 3, 1).cpu().numpy() * 255
            refs = rearrange(refs, "b (f c) h w -> b c h (f w)", f=5)
            ref_w = metadata["width"]
            ref_h = refs.shape[2] * ref_w // refs.shape[3]
            refs = F.interpolate(refs, size=(ref_h, ref_w), mode="bilinear")
            refs = refs.permute(0, 2, 3, 1).cpu().numpy() * 255
            refs = refs.astype(np.uint8)
            if add_stats:
                saliency = rearrange(saliency, "b (f c) h w -> b c h (f w)", f=5)
                saliency = F.interpolate(saliency, size=(ref_h, ref_w), mode="bilinear")
                saliency = saliency.permute(0, 2, 3, 1).cpu().numpy() * 255
                saliency = saliency.astype(np.uint8)
                image_saliency = image_saliency.permute(0, 2, 3, 1).cpu().numpy() * 255
                image_saliency = image_saliency.astype(np.uint8)
            for i in range(pred.shape[0]):
                out_face.write(pred[i].astype(np.uint8))
                img_idx = img_idxes[i]
                frame = frames[img_idx].copy()
                bbox = bboxes[img_idx]
                x1, y1, x2, y2, _ = bbox
                frame[y1:y2, x1:x2] = cv2.resize(pred[i], (x2 - x1, y2 - y1))
                if add_stats:
                    frame[0 : (y2 - y1) // 2, 0 : (x2 - x1) // 2] = cv2.resize(
                        image_saliency[i], ((x2 - x1) // 2, (y2 - y1) // 2)
                    )
                    # put lips region on top-right corner
                    frame[0 : (y2 - y1) // 2, -(x2 - x1) // 2 :] = cv2.resize(
                        lips_region[i], ((x2 - x1) // 2, (y2 - y1) // 2)
                    )
                    # put input image on bottom-right corner
                    frame[-(y2 - y1) // 2 :, -(x2 - x1) // 2 :] = cv2.resize(
                        input_image[i], ((x2 - x1) // 2, (y2 - y1) // 2)
                    )
                    # draw grid lines on refs[i] and saliency[i]
                if not add_stats:
                    frame = np.concatenate([frame, refs[i]], axis=0)
                else:
                    frame = np.concatenate([frame, refs[i], saliency[i]], axis=0)
                    # write max and min numbers audio saliency to frame at top right corner Use width from metadata
                    cv2.putText(
                        frame,
                        f"max: {max_aud_saliency[i]:.2f}, min: {min_aud_saliency[i]:.2f}, mean: {mean_aud_saliency[i]:.2f}",
                        (10, (x2 - x1) // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )
                pred_path = dst / video_path.stem / "preds" / f"frame_{img_idx}.png"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(pred_path), frame)
                out.write(frame)
        out_face.release()
        out.release()
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


def predict_ref_influencing_lips(model, image, ref_img, audio_embed, mask_dim, device, area="mouth"):
    image = image.to(device)
    ref_img = ref_img.to(device)
    audio_embed = audio_embed.to(device)
    ref_img.requires_grad = True
    image.requires_grad = True
    audio_embed.requires_grad = True
    pred, input_image = model(image, ref_img, audio_embed, mask_dim=mask_dim)
    # find lips region in the predicted image
    if area == "mouth":
        lips_region = model.segface.get_lips_region(pred).detach()
    else:
        lips_region = torch.ones_like(pred)
    if lips_region.max() == 1:
        loss = (pred * (lips_region == 1)).sum()
    else:
        loss = (pred * (lips_region > 0)).sum()
    loss.backward()
    # Process reference image saliency
    saliency = ref_img.grad.detach().abs()
    saliency = F.avg_pool2d(saliency, kernel_size=5, stride=1, padding=2)

    # Apply non-linear transformation to enhance contrast
    # This will suppress low values and enhance high values
    gamma = 2.0  # Adjust this value to control the contrast
    threshold = 0.1  # Adjust this to control what's considered "close to zero"

    # Normalize to [0, 1] first
    # for i in range(5):
    #     saliency[:, 3*i:3*i+3, :, :] = saliency[:, 3*i:3*i+3, :, :] / (saliency[:, 3*i:3*i+3, :, :].max() + 1e-8)
    saliency = saliency / (saliency.max() + 1e-8)
    # Apply threshold and gamma correction
    saliency = torch.where(saliency < threshold, torch.zeros_like(saliency), torch.pow(saliency, gamma))

    # Process input image saliency similarly
    image_saliency = image.grad.detach().abs()
    image_saliency = F.avg_pool2d(image_saliency, kernel_size=5, stride=1, padding=2)
    image_saliency = image_saliency / (image_saliency.max() + 1e-8)
    image_saliency = torch.where(
        image_saliency < threshold, torch.zeros_like(image_saliency), torch.pow(image_saliency, gamma)
    )

    # get max and min of audio saliency round up to 2 decimal places
    audio_saliency = audio_embed.grad.detach().abs()
    # audio_saliency = audio_saliency.round(2)
    max_audio_saliency = audio_saliency.max(1).values.max(1).values.max(1).values
    min_audio_saliency = audio_saliency.min(1).values.min(1).values.min(1).values
    mean_audio_saliency = audio_saliency.mean(1).mean(1).mean(1)
    return (
        pred,
        saliency,
        image_saliency,
        max_audio_saliency,
        min_audio_saliency,
        mean_audio_saliency,
        lips_region.repeat(1, 3, 1, 1),
        input_image,
    )


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
