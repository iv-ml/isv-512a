import os
import random
from pathlib import Path
from typing import Any

import fastcore.all as fc
import numpy as np
import torch
import torchvision
from prefetch_generator import BackgroundGenerator
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import ColorJitter

from lipsync.utils import load_json


def choose_far_number(numbers, excluded_number):
    # Remove the excluded number from the list
    available_numbers = [n for n in numbers if n != excluded_number]

    if not available_numbers:
        return None  # Return None if no numbers are available

    # Calculate weights based on the distance
    weights = [abs(n - excluded_number) for n in available_numbers]

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Choose a number based on the weights
    chosen_number = random.choices(available_numbers, weights=normalized_weights, k=1)[0]

    return chosen_number


def get_negative_pair(clips: list[dict[str, Any]], source_anchor: int) -> int:
    is_silent = clips[source_anchor].get("silence")
    if is_silent:
        not_silent_clips = [i for i in range(len(clips)) if not clips[i]["silence"] and i != source_anchor]
        return random.choice(not_silent_clips)
    else:
        return choose_far_number(list(range(len(clips))), source_anchor)


class SyncNetDataset(Dataset):
    def __init__(
        self,
        data_root,
        ds_name,
        audio_features_name,
        json_loc,
        augment_num=1,
        train_val="train",
        verbose=True,
        bad_clips=None,
    ):
        super(SyncNetDataset, self).__init__()
        self.data_root = fc.Path(data_root)
        self.agument = augment_num
        self.ds_name = ds_name
        self.audio_features_name = audio_features_name
        self.json_loc = json_loc
        self.train_val = train_val
        self.records = load_json(self.json_loc)
        self.bad_clips = bad_clips
        if self.bad_clips is not None:
            # read a txt file
            total_clips = len(self.records.keys())
            remove_clips = [
                i.rsplit("/")[-1]
                for i in fc.Path(bad_clips).read_text().splitlines()
                if i.rsplit("/")[0] == self.ds_name
            ]
            self.records = {k: v for k, v in self.records.items() if k not in remove_clips}
            print(f"Removed {total_clips - len(self.records.keys())} bad clips")

        if self.train_val == "val":
            self.agument = 1
            bins = []
            for k, v in self.records.items():
                clip_bins: list[dict[str, Any]] = v["clips"]
                bins.extend([{**bin, "clip_id": k} for bin in clip_bins])
            self.videos = bins
        else:
            # Filter videos: keep only those with at least 2 clips AND at least 1 non-silent clip
            self.videos = fc.L(
                [
                    video_id
                    for video_id in self.records.keys()
                    if len(self.records[video_id]["clips"]) > 2
                    and any(not clip["silence"] for clip in self.records[video_id]["clips"])
                ]
            )
        if verbose:
            print(f"Total videos/bins after filtering: {len(self.videos)}")
            print(f"Filtered out {len(self.records) - len(self.videos)} videos/bins with only silent clips")

        # we are shuffling for randomness
        self.videos = self.videos * self.agument
        if self.train_val == "train":
            random.shuffle(self.videos)

        self.train_val = train_val
        self.tf1 = ColorJitter(brightness=0.5, contrast=0.5, saturation=None, hue=None)
        self.tf2 = ColorJitter(brightness=None, contrast=None, saturation=0.5, hue=0.5)

    def __getitem__(self, index):
        target_label = 1
        if self.train_val == "train":
            video_name = self.videos[index]
            data = self.records[video_name]["clips"]
            source_anchor = random.sample(range(len(data)), 1)[0]
            audio_anchor = source_anchor
            # if self.neg_pair and (self.train_val == "train"):
            # get negative pair
            target_label = np.random.choice([0, 1])
            # if np.random.randn(1) > 0:
            if target_label == 0:
                audio_anchor = get_negative_pair(data, source_anchor)
                if audio_anchor is None:
                    target_label = 1
                    audio_anchor = source_anchor
            source_bin = data[source_anchor]
            audio_bin = data[audio_anchor]
        else:
            source_bin = self.videos[index]
            audio_bin = source_bin
            video_name = source_bin["clip_id"]
            silence = source_bin["silence"]

        # Audio features
        audio_path = self.data_root / "audio" / self.ds_name / (video_name + ".safetensors")
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
        wvf = load_file(audio_path)["audio_embedding"]
        deep_speech_list = wvf[audio_bin["start_audio"] : audio_bin["end_audio"] + 1]
        # image features
        img_locs = source_bin["images"]
        img_folder = source_bin["folder_name"]
        images = []
        start_frame = 0  # 9 we need to pick up 5 = [2, 3, 4, 5, 6]

        for source_frame_index in range(start_frame, start_frame + 5):
            ## load source clip
            loc = (
                self.data_root
                / "video_hallo_512"
                / self.ds_name
                / video_name
                / img_folder
                / img_locs[source_frame_index]
            )
            # Read using torchvision
            source_image_data = torchvision.io.read_image(str(loc), mode=torchvision.io.ImageReadMode.RGB)
            source_image_data = source_image_data[:, 224 : 480 - 96, 128:384]
            source_image_data = torchvision.transforms.functional.resize(source_image_data, (80, 128))
            images.append(source_image_data)

        source_lip = torch.stack(images)
        if self.train_val == "train":
            if np.random.randn(1) > 0.5:
                source_lip = self.tf1(source_lip)
            if np.random.randn(1) > 0.5:
                source_lip = self.tf2(source_lip)
        source_lip = source_lip / 255.0

        deep_speech_clip = torch.as_tensor(deep_speech_list).float()
        target = torch.as_tensor([target_label])
        if self.train_val == "val":
            silence = torch.as_tensor([silence])
            return (
                source_lip,
                deep_speech_clip,
                silence,
                f"{self.ds_name}/{source_bin['clip_id']}_{source_bin['folder_name']}",
            )
        else:
            return source_lip, deep_speech_clip, target

    def __len__(self):
        return len(self.videos)


class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=1)


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
    train_data = [
        SyncNetDataset(
            data_root=d.data_root,
            ds_name=d.ds_name,
            json_loc=d.train_json,
            audio_features_name=d.audio_features_name,
            augment_num=opt.augment_num[opt.stage],
            train_val="train",
            bad_clips=opt.bad_clips,
        )
        for d in opt.data
    ]
    train_data = CombinedDataset(train_data)

    val_data = [
        SyncNetDataset(
            data_root=d.data_root,
            ds_name=d.ds_name,
            json_loc=d.val_json,
            audio_features_name=d.audio_features_name,
            augment_num=opt.augment_num[opt.stage],
            train_val="train",
            bad_clips=opt.bad_clips,
        )
        for d in opt.data
        if d.val_json is not None
    ]
    val_data = CombinedDataset(val_data)

    train_data_loader = Union_Dataloader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
    )

    val_data_loader = Union_Dataloader(
        dataset=val_data,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=opt.num_workers,
    )
    return train_data_loader, val_data_loader


if __name__ == "__main__":
    from pathlib import Path

    df = SyncNetDataset(
        data_root=Path("/data/prakash_lipsync/"),
        ds_name="th_1kh_512",
        audio_features_name="fe_wav2vec12",
        json_loc="/data/prakash_lipsync/v1/th_1kh_512/train_clips_hallo_filtered.json",
        augment_num=10,
        train_val="train",
        bad_clips="scripts/syncnet/bad_clips.txt",
    )
    for i in range(len(df)):
        x, y, z = df[np.random.randint(0, len(df))]
        print(x.shape, y.shape, z)
        # use the below code to save images to disk and visualize the dataset output.
        xt = (x * 255).to(torch.uint8)
        for n, i in enumerate(xt):
            Path("temp").mkdir(exist_ok=True)
            torchvision.io.write_png(i, f"temp/img_{n}.png")
        break

    val_data_loader = Union_Dataloader(dataset=df, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    for n, ix in enumerate(val_data_loader):
        # print(n, [(i.cuda() * 2).shape for i in ix[:2]], len(ix[-1])) #ix[-1].sum())
        print(n, [(i.cuda() * 2).shape for i in ix], ix[-1].sum())
