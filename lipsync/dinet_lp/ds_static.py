# Works on the new data structures

import random

import fastcore.all as fc
import numpy as np
import torch
import torch.utils
import torchvision
from prefetch_generator import BackgroundGenerator
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import ColorJitter

from lipsync.utils import load_json


class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=1)


class ClipDataset(Dataset):
    def __init__(
        self,
        data_root,
        ds_name,
        audio_features_name,
        video_folder,
        json_loc,
        augment_num,
        train_val="train",
        silences=None,
    ):
        super(ClipDataset, self).__init__()
        self.data_root = fc.Path(data_root)
        self.agument = augment_num
        self.ds_name = ds_name
        self.audio_features_name = audio_features_name
        self.video_folder = video_folder
        self.json_loc = json_loc
        self.train_val = train_val
        self.silences = silences
        self.records = load_json(self.json_loc)
        # remove bad clips
        self.records = self.remove_bad_clips(self.records)
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

    def remove_bad_clips(self, records):
        # Create a copy to avoid modifying while iterating
        records_copy = records.copy()

        for video_name, video_data in records_copy.items():
            clips = video_data["clips"]
            if all(clip["silence"] for clip in clips):
                print(f"Deleting video {video_name} because all clips are silent")
                del records[video_name]

        return records

    def __getitem__(self, index):
        # we need to simplify this to choose silence 20% of the time.
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

        # get landmarks
        f_landmarks = np.stack(
            [(np.asarray(i["eye_center"]) + np.asarray(i["lip_center"])) / 2 for i in source_image_path_list["lms"]]
        )
        f_landmarks = f_landmarks.astype(np.int32)
        landmarks = torch.ones((5, 1, 512, 512), dtype=torch.int32)
        for i, f_landmark in enumerate(f_landmarks):
            _, y = f_landmark
            landmarks[i, :, y - 32 : 512 - 32, 96 : 512 - 96] = 0

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
            available_indices.remove(source_anchor)
            reference_anchor_list = random.sample(available_indices, 5)

            for reference_anchor in reference_anchor_list:
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
                # TODO: apply transforms here.
                reference_frame_list.append(reference_image_data)

            reference_clip_list.append(torch.stack(reference_frame_list))

        source_clip = torch.stack(source_clip_list)
        reference_clip = torch.vstack(reference_clip_list)

        # stack them, apply transforms - resize, etc and then un-stack them again similart to how we have done in syncnet
        all_clips = torch.cat([source_clip, reference_clip], 0)
        if self.train_val == "train":
            if np.random.randn(1) > 0.5:
                all_clips = self.tf1(all_clips)
            else:
                all_clips = self.tf2(all_clips)

        all_clips = all_clips / 255.0

        source_clip = all_clips[:5, ...]
        # reference clip is 25x3x256x256 and we need to unstack 5x5x3x256x256 and then concat them.
        reference_clip = all_clips[5:, ...]
        reference_clip = torch.cat(torch.split(reference_clip, 5, dim=0), 1)
        deep_speech_clip = torch.stack(deep_speech_list)

        deep_speech_syncnet = wvf[source_image_path_list["start_audio"] : source_image_path_list["end_audio"] + 1, ...]
        if self.silences:
            return (
                source_clip,
                landmarks,
                reference_clip,
                deep_speech_clip,
                deep_speech_syncnet,
                torch.tensor(silence_anchor),
            )
        else:
            return (
                source_clip,
                landmarks,
                reference_clip,
                deep_speech_clip,
                deep_speech_syncnet,
                torch.tensor(0),  # A dummy value and should never be used.
            )

    def __len__(self):
        return len(self.videos)


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
        ClipDataset(
            data_root=d.data_root,
            json_loc=d.train_json,
            ds_name=d.ds_name,
            audio_features_name=d.audio_features_name,
            video_folder=d.video_folder,
            augment_num=opt.augment_num,
            train_val="train",
            silences=d.silences,
        )
        for d in opt.data
    ]
    train_data = CombinedDataset(train_data)

    val_data = [
        ClipDataset(
            data_root=d.data_root,
            json_loc=d.val_json,
            ds_name=d.ds_name,
            audio_features_name=d.audio_features_name,
            video_folder=d.video_folder,
            augment_num=opt.augment_num,
            train_val="val",
            silences=True,  # this will make sure silence outputs are present in val set.
        )
        for d in opt.data
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
        drop_last=True,
        num_workers=opt.num_workers,
    )
    return train_data_loader, val_data_loader


if __name__ == "__main__":
    df = ClipDataset(
        "/data/prakash_lipsync",
        ds_name="iv_recording",
        audio_features_name="fe_wav2vec12",
        video_folder="video_hallo_512",
        json_loc="/data/prakash_lipsync/v1/iv_recording/train_hallo_512_lm.json",
        augment_num=1,
        train_val="train",
        silences=True,  # "data/silence_bins.csv",
    )
    x = df[np.random.randint(0, len(df))]
    print([i.shape for i in x])
    # visualize input
    x2 = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(x[0], nrow=5))
    x2.save("temp/source_lp.png")

    xmask = x[1] * x[0]
    x2 = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(xmask, nrow=5))
    x2.save("temp/mask_lp.png")

    # visualize references
    reference_clip = x[2]
    # convert 5, 15, 256x256 to 25x3x256x256
    reference_clip = torch.cat(torch.split(reference_clip, 3, dim=1), 0)
    x1 = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(reference_clip, nrow=5))
    x1.save("temp/references_lp.png")
    # breakpoint()

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
