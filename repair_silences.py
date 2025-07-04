from pathlib import Path

import fastcore.all as fc
import numpy as np
import typer
from tqdm import tqdm


def realign_landmarks_to_size(landmarks, bins):
    def realign_landmarks(lm, x1, y1, x2, y2):
        lm[:, 0] = (lm[:, 0] - x1) / (x2 - x1) * 512
        lm[:, 1] = (lm[:, 1] - y1) / (y2 - y1) * 512
        return lm

    new_landmarks = []
    for batch in bins:
        batch_landmarks = []
        for b in batch:
            _, frame_number, x1, y1, x2, y2 = b
            lm = landmarks[frame_number, :, :3].copy()
            lm = realign_landmarks(lm, x1, y1, x2, y2)
            batch_landmarks.append(lm)
        new_landmarks.append(np.stack(batch_landmarks))
    new_landmarks = np.stack(new_landmarks)
    assert (
        new_landmarks.shape[0] == bins.shape[0]
    ), f"New landmarks shape {new_landmarks.shape} does not match bins shape {bins.shape}"
    return new_landmarks  # n_bins x n_frames x n_landmarks x 3 (x, y, z)


def normalize_and_scale_landmarks(landmarks):
    NOSE_TOP = 10
    NOSE_BOTTOM = 1
    nose_height = np.linalg.norm(landmarks[:, NOSE_TOP, :] - landmarks[:, NOSE_BOTTOM, :], axis=1)
    landmarks = landmarks / nose_height[:, None, None] * 100.0
    return landmarks


def calculate_lip_distance(landmarks):
    LIP_TOP = 13
    LIP_BOTTOM = 14
    normalized_landmarks = normalize_and_scale_landmarks(landmarks)
    lip_distances = np.linalg.norm(normalized_landmarks[:, LIP_TOP, :] - normalized_landmarks[:, LIP_BOTTOM, :], axis=1)
    return lip_distances


def main(
    json_root: str = typer.Argument("/data/.pipeliner_cache/jsons", help="Path to the JSON files"),
    ds_names: str = typer.Argument("iv_recording,iv_recording_v2", help="Comma separated list of dataset names"),
    train_val: str = typer.Argument("train", help="train or val"),
):
    from lipsync.utils import load_json, save_json

    # bins_root = "/data/.pipeliner_cache/video/bins"
    # check if json file exists
    for ds_name in ds_names.split(","):
        for train_val in ["train", "val"]:
            if not Path(f"{json_root}/{ds_name}/{ds_name}_{train_val}.json").exists():
                print(f"JSON file {json_root}/{ds_name}/{ds_name}_{train_val}.json does not exist")
                continue
            ds = load_json(f"{json_root}/{ds_name}/{ds_name}_{train_val}.json")
            # root = fc.Path(f"/data/prakash_lipsync/video_hallo_256_16/{ds_name}")
            videos = fc.L(ds.keys())
            for video in tqdm(videos, desc=f"Processing {ds_name} {train_val}"):
                data = ds[video]["clips"]
                for n, clip in enumerate(data):
                    ds[video]["clips"][n]["silence_filtered"] = 0
                    ds[video]["clips"][n]["lip_distances"] = []

                # frame_numbers = [int(i.rsplit(".")[0]) for i in images]
                # ld = [lip_distance[np.where(all_frames == i)[0][0]] for i in frame_numbers]
                landmarks_root = fc.Path("/data/.pipeliner_cache/mp_lms")
                landmarks_path = landmarks_root / ds_name / f"{video}.npy"
                landmarks = np.load(landmarks_path)
                # bins_path = fc.Path(f"{bins_root}/{ds_name}/") / f"{video}.npy"
                # bins = np.load(bins_path)
                # realign_new_landmarks = realign_landmarks_to_size(landmarks, bins)
                # bin_index_map = {}
                new_clips = []
                deleted = 0
                lip_distance = calculate_lip_distance(landmarks)
                for n, bin in enumerate(data):
                    image_ids = [int(i.rsplit(".")[0]) for i in bin["images"]]
                    ld = lip_distance[image_ids].tolist()
                    audio_silence = bin["silence"]
                    video_silence = all([l < 0.1 for l in ld])
                    if audio_silence and not video_silence:
                        deleted += 1
                        continue
                    bin["silence_filtered"] = audio_silence and video_silence
                    bin["lip_distances"] = ld
                    new_clips.append(bin)
                ds[video]["clips"] = new_clips
                if deleted > 0:
                    print(f"Deleted {deleted} clips from {video}")

            save_json(f"{json_root}/{ds_name}/{train_val}_silence_filtered_new.json", ds)


if __name__ == "__main__":
    # main()
    typer.run(main)
