# Works on the new data structures

import random
from typing import List, Optional, Tuple

import fastcore.all as fc
import numpy as np
import torch
import torch.utils
import torchvision
import torchvision.transforms.v2.functional as F
from prefetch_generator import BackgroundGenerator
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import (
    ColorJitter,
    RandomHorizontalFlip,
    RandomPerspective,
    Transform,
)

from lipsync.utils import load_json

torch.random.manual_seed(42)
np.random.seed(42)
random.seed(42)


def augment_videos(videos, vl, N):
    # I want increase the increase len of the videos by N times and want to agument based on weights of vl (more weight more likely to be picked)
    # Also want all videos to be picked atleast once
    x = fc.L([i for i in range(len(videos)) for _ in range(int(N * vl[i]))])
    new_videos = fc.L([videos[i] for i in x])
    return new_videos + videos  # makes sure that all videos are picked atleast once


class RandomZoomIn(Transform):
    def __init__(
        self,
        scale: Tuple[float, float] = (1.1, 1.5),
        interpolation: Optional[F.InterpolationMode] = F.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = None,
    ):
        """
        Randomly zooms in on the center of the image.

        Args:
            scale: Tuple of (min_scale, max_scale). The image will be zoomed in by a
                  random factor between these values. E.g. (1.1, 1.5) means zoom between
                  110% and 150%.
            interpolation: Interpolation mode for resizing
            antialias: Whether to use antialiasing
        """
        super().__init__()
        self.scale = scale
        self.interpolation = interpolation
        self.antialias = antialias

        if not (isinstance(scale, tuple) and len(scale) == 2 and all(isinstance(s, (int, float)) for s in scale)):
            raise ValueError("scale must be a tuple of two numbers")
        if not (1.0 <= scale[0] <= scale[1]):
            raise ValueError("scale values must be >= 1.0 and min_scale <= max_scale")

    def _get_params(self, flat_inputs: List[torch.Tensor]) -> dict:
        # Generate random zoom factor
        scale_factor = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
        return {"scale_factor": scale_factor}

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Args:
            inpt: Input tensor image of shape (C, H, W)
            params: Dictionary containing the scale_factor
        Returns:
            Zoomed in tensor image of shape (C, H, W)
        """
        if not isinstance(inpt, torch.Tensor):
            raise TypeError(f"Input should be a tensor, got {type(inpt)}")

        scale_factor = params["scale_factor"]

        # Calculate new dimensions
        if len(inpt.shape) == 4:
            _, _, h, w = inpt.shape
        else:
            _, h, w = inpt.shape
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        # First resize the image to larger dimensions
        resized = F.resize(inpt, [new_h, new_w], interpolation=self.interpolation, antialias=self.antialias)

        # Calculate center crop dimensions to get back to original size
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2

        # Perform center crop
        if len(inpt.shape) == 4:
            zoomed = resized[:, :, start_h : start_h + h, start_w : start_w + w]
        else:
            zoomed = resized[:, start_h : start_h + h, start_w : start_w + w]

        return zoomed


class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=1)


class ClipDataset(Dataset):
    def __init__(
        self,
        data_root,
        ds_name,
        video_folder,
        landmarks_folder,
        bbox_folder,
        json_loc,
        augment_num,
        upscale=1,
        reference_frames_transforms=False,
        train_val="train",
        silences=None,
        weighted_augment=False,
        length=None,
    ):
        super(ClipDataset, self).__init__()
        self.data_root = fc.Path(data_root)
        self.landmarks_folder = landmarks_folder
        self.bbox_folder = bbox_folder
        self.agument = augment_num
        self.ds_name = ds_name
        self.video_folder = video_folder
        self.json_loc = json_loc
        self.train_val = train_val
        self.silences = silences
        self.upscale = upscale
        self.reference_frames_transforms = reference_frames_transforms
        self.records = load_json(self.json_loc)
        self.length = length
        # remove clips which havs less than 6 bins
        videos_to_remove = [k for k, v in self.records.items() if len(v["clips"]) < 6]
        print(f"remoing a a total of {len(videos_to_remove)} from {len(self.records.keys())}")
        for v in videos_to_remove:
            del self.records[v]
        # remove bad clips
        self.records = self.remove_bad_clips(self.records)
        if weighted_augment:
            vl = fc.L([len(self.records[v]["clips"]) for v in self.records.keys()])
            max_vl = np.percentile(vl, 90)
            vl = [round(i / max_vl, 2) for i in vl]
            self.videos = augment_videos(fc.L(self.records.keys()), vl, self.agument)
        else:
            self.videos = fc.L(self.records.keys()) * self.agument
        if self.train_val == "train":
            random.shuffle(self.videos)
        else:
            records = []
            for i in self.records.keys():
                for j in range(len(self.records[i]["clips"])):
                    records.append([i, j])
            self.ds = fc.L(records)

        # lets add transforms similar to syncnet training: we need to apply same transforms to source and reference images
        self.tf1 = ColorJitter(brightness=0.5, contrast=0.5, saturation=None, hue=None)
        self.tf2 = ColorJitter(brightness=None, contrast=None, saturation=0.5, hue=0.5)
        # self.cutout = cutout_torch(128*self.upscale, 0.8, False)

    def remove_bad_clips(self, records):
        # Create a copy to avoid modifying while iterating
        records_copy = records.copy()

        for video_name, video_data in records_copy.items():
            clips = video_data["clips"]
            if all(clip["silence"] for clip in clips):
                print(f"Deleting video {video_name} because all clips are silent")
                del records[video_name]

        return records

    def __getitem__(self, _):
        # we need to simplify this to choose silence 20% of the time.
        index = random.randint(0, len(self.videos) - 1)
        silence_anchor = 0
        if self.train_val == "train":
            video_name = self.videos[index]
            data = self.records[video_name]["clips"]
            if self.silences:
                silence_anchor = 1
                silences = [i["silence"] for i in data]
                if np.random.rand() < 0.2:
                    # get index of silent clip
                    silent_index = [i for i, j in enumerate(silences) if j]
                    # there can be chace that there are no silent index only
                    if len(silent_index) == 0:
                        source_anchor = random.sample(range(len(data)), 1)[0]
                        silence_anchor = 0
                    else:
                        source_anchor = random.sample(silent_index, 1)[0]
                else:
                    # we made sure that there is atleast one non-silent clip in the video using remove_bad_clips
                    non_silent_index = [i for i, j in enumerate(silences) if not j]
                    source_anchor = random.sample(non_silent_index, 1)[0]
                    silence_anchor = 0
            else:
                source_anchor = random.sample(range(len(data)), 1)[0]
        else:
            video_name, source_anchor = self.ds[index]
            data = self.records[video_name]["clips"]
            silence_anchor = 1 if data[source_anchor]["silence"] else 0

        # Audio features
        audio_path = self.data_root / "audio" / self.ds_name / (video_name + ".safetensors")
        wvf = load_file(audio_path)["audio_embedding"]
        source_image_path_list = data[source_anchor]
        source_clip_list = []
        deep_speech_list = []
        reference_clip_list = []
        # source [2, 7] for each img we get 5 reference images and they come from 5 differents bins.
        # To we are going to 9
        start_frame = 0
        for source_frame_index in range(start_frame, 5):
            # frame

            source_loc = (
                self.data_root
                / self.video_folder
                / self.ds_name
                / video_name
                / source_image_path_list["folder_name"]
                / source_image_path_list["images"][source_frame_index]
            )
            source_image_data = torchvision.io.read_image(source_loc, mode=torchvision.io.ImageReadMode.RGB)
            source_image_data = source_image_data

            source_clip_list.append(source_image_data)
            # Source audio
            start_audio = 0
            end_audio = 5
            source_audio_data = wvf[
                source_image_path_list["start_audio"] + source_frame_index + start_audio : source_image_path_list[
                    "start_audio"
                ]
                + source_frame_index
                + end_audio,
                ...,
            ]
            deep_speech_list.append(source_audio_data)

            ## load reference images
            reference_frame_list = []
            available_indices = list(range(len(data)))
            # if source anchor is silent, then we need to pick non silent bins only
            if silence_anchor == 1:
                available_indices = [i for i in available_indices if not data[i]["silence"]]
            if source_anchor in available_indices:
                available_indices.remove(source_anchor)
            if (len(available_indices) > 0) & (len(available_indices) < 5):
                available_indices = available_indices * 5
            else:
                available_indices = list(range(len(data)))
                available_indices.remove(source_anchor)
            reference_anchor_list = random.sample(available_indices, 5)
            for _, reference_anchor in enumerate(reference_anchor_list):
                reference_frame_path_list = data[reference_anchor]
                reference_frame_index = random.sample(range(len(reference_frame_path_list["images"])), 1)[0]
                # frame
                reference_loc = (
                    self.data_root
                    / self.video_folder
                    / self.ds_name
                    / video_name
                    / reference_frame_path_list["folder_name"]
                    / reference_frame_path_list["images"][reference_frame_index]
                )

                reference_image_data = torchvision.io.read_image(reference_loc, mode=torchvision.io.ImageReadMode.RGB)
                if self.reference_frames_transforms:
                    if self.train_val == "train":
                        reference_image_data = RandomHorizontalFlip(p=0.5)(reference_image_data)
                        # reference_image_data = RandomZoomOut(side_range=(1, 2))(reference_image_data)
                        # reference_image_data = RandomRotation(degrees=(-20, 20))(reference_image_data)
                        reference_image_data = RandomPerspective(distortion_scale=0.2, p=0.5)(reference_image_data)
                        reference_image_data = RandomZoomIn(scale=(1, 1.5))(reference_image_data)
                        # reference_image_data = RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.8, 1.2))(reference_image_data)

                reference_frame_list.append(reference_image_data)

            reference_clip_list.append(torch.stack(reference_frame_list))

        source_clip = torch.stack(source_clip_list)
        if self.train_val == "train":
            source_clip = RandomZoomIn(scale=(1, 1.05))(source_clip)
        reference_clip = torch.vstack(reference_clip_list)

        # stack them, apply transforms - resize, etc and then un-stack them again similart to how we have done in syncnet
        all_clips = torch.cat([source_clip, reference_clip], 0)
        if self.train_val == "train":
            if np.random.randn(1) > 0.5:
                all_clips = self.tf1(all_clips)
            if np.random.randn(1) > 0.5:
                all_clips = self.tf2(all_clips)
            all_clips = RandomHorizontalFlip(p=0.5)(all_clips)

        if all_clips.shape[2] != self.upscale * 256:
            raise ValueError(f"Image size {all_clips.shape[2]} does not match expected size {self.upscale * 256}")
            # all_clips = torchvision.transforms.functional.resize(all_clips, (self.upscale * 256, self.upscale * 256))
        all_clips = all_clips / 255.0

        source_clip = all_clips[:5, ...]
        # reference clip is 25x3x256x256 and we need to unstack 5x5x3x256x256 and then concat them.
        reference_clip = all_clips[5:, ...]
        reference_clip = F.resize(reference_clip, (256, 256))  # resize references to 256x256
        reference_clip = torch.cat(torch.split(reference_clip, 5, dim=0), 1)

        # if self.train_val == "train":
        #     source_clip = torchvision.transforms.RandomHorizontalFlip(p=0.5)(source_clip)

        deep_speech_clip = torch.stack(deep_speech_list)

        deep_speech_syncnet = wvf[source_image_path_list["start_audio"] : source_image_path_list["end_audio"] + 1, ...]
        if self.silences:
            return (
                source_clip,
                reference_clip,
                deep_speech_clip,
                deep_speech_syncnet,
                torch.tensor(silence_anchor),
            )
        else:
            return (
                source_clip,
                reference_clip,
                deep_speech_clip,
                deep_speech_syncnet,
                torch.tensor(0),  # A dummy value and should never be used.
            )

    def __len__(self):
        if self.length is not None:
            return self.length
        if self.train_val == "train":
            return 16 * 1500  # 16 * 1500
        else:
            return 16 * 100


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.combined_data = []
        for dataset in datasets:
            self.combined_data.extend([(i, dataset) for i in range(len(dataset))])

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        original_idx, dataset = self.combined_data[idx]
        return dataset[original_idx]


def load_concat_data(opt):
    num_datasets = len([d for d in opt.data if d.train_json is not None])
    train_data = [
        ClipDataset(
            data_root=d.data_root,
            json_loc=d.train_json,
            ds_name=d.ds_name,
            audio_features_name=d.audio_features_name,
            video_folder=d.video_folder,
            augment_num=opt.augment_num,
            upscale=opt.upscale,
            train_val="train",
            silences=d.silences,
            reference_frames_transforms=d.reference_frames_transforms,
            length=1000 * opt.batch_size * opt.devices * opt.nodes // num_datasets,
        )
        for d in opt.data
        if d.train_json is not None
    ]
    train_data = CombinedDataset(train_data)
    num_val_datasets = len([d for d in opt.data if d.val_json is not None])
    val_data = [
        ClipDataset(
            data_root=d.data_root,
            json_loc=d.val_json,
            ds_name=d.ds_name,
            audio_features_name=d.audio_features_name,
            video_folder=d.video_folder,
            augment_num=opt.augment_num,
            upscale=opt.upscale,
            train_val="val",
            silences=d.silences,  # this will make sure silence outputs are present in val set.
            reference_frames_transforms=d.reference_frames_transforms,
            length=100 * opt.batch_size * opt.devices * opt.nodes // num_val_datasets,
        )
        for d in opt.data
        if d.val_json is not None
    ]
    val_data = CombinedDataset(val_data)

    train_data_loader = Union_Dataloader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=opt.num_workers,
    )

    val_data_loader = Union_Dataloader(
        dataset=val_data,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=opt.num_workers,
    )
    return train_data_loader, val_data_loader


if __name__ == "__main__":
    df = ClipDataset(
        "/data/lipsync_768_data",
        ds_name="iv_recording_v2",
        audio_features_name="fe_wav2vec12",
        video_folder="video_hallo_512",
        json_loc="/data/lipsync_768_data/v1/iv_recording_v2/train_hallo.json",
        augment_num=4,
        upscale=3,
        train_val="train",
        silences=True,  # "data/silence_bins.csv",
        reference_frames_transforms=False,
        weighted_augment=False,  # true 7584, false we have 15432 for augment 4
    )
    print("total videos", len(df))
    x = df[np.random.randint(0, len(df))]
    print([i.shape for i in x])
    # visualize input
    x2 = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(x[0], nrow=9))
    x2.save("temp/source.png")

    # visualize references
    reference_clip = x[1]
    # convert 5, 15, 256x256 to 25x3x256x256
    reference_clip = torch.cat(torch.split(reference_clip, 3, dim=1), 0)
    x1 = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(reference_clip, nrow=5))
    x1.save("temp/references.png")

    val_data_loader = Union_Dataloader(dataset=df, batch_size=8, shuffle=False, drop_last=False, num_workers=8)
    import time

    start = time.time()
    count = 0
    silence_count = 0
    for n, ix in enumerate(val_data_loader):
        count += len(ix[0])
        silence_count += ix[-1].sum().item()
        print(n, [i.shape for i in ix], ix[-1].sum(), f"{(time.time() - start)*1000:.2f}ms")
    print(f"total_time: {(time.time() - start)*1000:.2f}ms")
    print(f"count: {count}, silence_count: {silence_count}, percentage: {silence_count/count}")
