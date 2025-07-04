import random
from pathlib import Path
from typing import Any

import fastcore.all as fc
import numpy as np
import torch
import torchvision
from einops import rearrange
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import ColorJitter

from lipsync.utils import load_json


def augment_videos(videos, vl, N):
    # I want increase the increase len of the videos by N times and want to agument based on weights of vl (more weight more likely to be picked)
    # Also want all videos to be picked atleast once
    x = fc.L([i for i in range(len(videos)) for _ in range(int(N * vl[i]))])
    new_videos = fc.L([videos[i] for i in x])
    return new_videos + videos  # makes sure that all videos are picked atleast once


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
        weighted_augment=True,
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
        if weighted_augment:
            vl = fc.L([len(self.records[v]["clips"]) for v in self.records.keys()])
            max_vl = np.percentile(vl, 90)
            vl = [round(i / max_vl, 2) for i in vl]
            self.videos = augment_videos(fc.L(self.records.keys()), vl, self.agument)
        else:
            self.videos = self.videos * self.agument
        if self.train_val == "train":
            random.shuffle(self.videos)

        self.train_val = train_val
        self.tf1 = ColorJitter(brightness=0.5, contrast=0.5, saturation=None, hue=None)
        self.tf2 = ColorJitter(brightness=None, contrast=None, saturation=0.5, hue=0.5)

    def __getitem__(self, index):
        # lets pick from video index silence with 20% prob
        if self.train_val == "train":
            video_name = self.videos[index]
            data = self.records[video_name]["clips"]
            silence = np.asarray([i["silence"] for i in data])
            silence_idx = np.where(silence == 1)[0]
            non_silence_idx = np.where(silence == 0)[0]
            if np.random.randint(10) / 10 > 0.5:
                idx = non_silence_idx
            else:
                idx = silence_idx
            idx = range(len(data)) if len(idx) == 0 else idx.tolist()
            source_anchor = random.sample(idx, 1)[0]
            source_bin = data[source_anchor]
            silence = source_bin["silence"]
        else:
            source_bin = self.videos[index]
            video_name = source_bin["clip_id"]
            silence = source_bin["silence"]

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
            source_image_data = torchvision.transforms.functional.resize(source_image_data, (256, 256))
            images.append(source_image_data)
        source_lip = torch.stack(images)
        if self.train_val == "train":
            if np.random.randn(1) > 0.5:
                source_lip = self.tf1(source_lip)
            if np.random.randn(1) > 0.5:
                source_lip = self.tf2(source_lip)
        # rearrange the source_lip to 5x3x256x256 to 15x256x256 using einops
        source_lip = rearrange(source_lip, "b c h w -> (b c) h w")
        source_lip = source_lip / 255.0

        return source_lip, torch.as_tensor([silence])

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
        ds_name="hdtf",
        audio_features_name="fe_wav2vec12",
        json_loc="/data/prakash_lipsync/v2/hdtf/train_hallo.json",
        augment_num=10,
        train_val="train",
        bad_clips="scripts/syncnet/bad_clips.txt",
    )
    for i in range(len(df)):
        x, y = df[np.random.randint(0, len(df))]
        print(x.shape, y.shape, y)
        break
        # use the below code to save images to disk and visualize the dataset output.
        # xt = (x * 255).to(torch.uint8)
        # for n, i in enumerate(xt):
        #     Path("temp").mkdir(exist_ok=True)
        #     torchvision.io.write_png(i, f"temp/img_{n}.png")
        # break

    val_data_loader = Union_Dataloader(dataset=df, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    for n, ix in enumerate(val_data_loader):
        print(n, ix[0].shape, ix[1].shape, ix[1].sum())
