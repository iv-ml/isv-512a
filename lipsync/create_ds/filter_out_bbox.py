## remove a bin if
## 1. landmarks outside the image
## 2. lip landmarks outside the image
## for each clip, remove it if it has less than 2 bins.

import fastcore.all as fc
import numpy as np
import typer

from lipsync.utils import load_json, save_json

LIP_IDS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_IDS = LEFT_EYE + RIGHT_EYE


def main(
    mp_lms: fc.Path = typer.Option("/data/mp_lms", help="Path to subclip landmarks"),
    bins_path: fc.Path = typer.Option("/data/prakash_lipsync/video_hallo_512/bins", help="Path to bins"),
    json_path: fc.Path = typer.Option(
        "/data/prakash_lipsync/v1/th_1kh_512/train_clips_hallo.json", help="Path to json files"
    ),
    ds_name: str = typer.Option("th_1kh_512", help="Name of the dataset"),
):
    landmarks_dir = mp_lms / ds_name
    bins_path = bins_path / ds_name

    mask = [64 * 2, 112 * 2, 192 * 2, 240 * 2]
    vis = False

    x = load_json(json_path)
    videos = fc.L(x.keys())
    total_bins = sum([len(x[video]["clips"]) for video in videos])
    print("total bins", total_bins)
    for video in videos:
        bins = x[video]["clips"]
        # load its landmarks
        landmarks = np.load(landmarks_dir / (video + ".npy"))
        bbox = np.load(bins_path / (video + ".npy"))
        new_bins = []
        for n, bin in enumerate(bins):
            images = bin["images"]
            keeps = []
            for m, img in enumerate(images):
                frame_number = int(img.rsplit(".")[0])
                bin_number = np.where(bbox[:, :, 1] == frame_number)[0]
                _, frame_number_2, x1, y1, x2, y2 = bbox[bin_number][0][m]
                lm = landmarks[frame_number_2, :, :2].copy()
                lm[:, 0] = (lm[:, 0] - x1) / (x2 - x1) * 512
                lm[:, 1] = (lm[:, 1] - y1) / (y2 - y1) * 512
                # check if the landmarks are within the image
                lip_lm = lm[LIP_IDS, :]
                if vis:
                    import matplotlib.pyplot as plt
                    from PIL import Image

                    im = Image.open(
                        fc.Path(f"/data/prakash_lipsync/video_hallo_512/{ds_name}/{video}/{bin['folder_name']}/{img}")
                    )
                    plt.figure(figsize=(10, 5))
                    plt.imshow(im)
                    # draw bbox using mask [x1, y1, x2, y2]
                    plt.plot(
                        [mask[0], mask[2], mask[2], mask[0], mask[0]],
                        [mask[1], mask[1], mask[3], mask[3], mask[1]],
                        c="g",
                    )
                    plt.scatter(lm[:, 0], lm[:, 1], c="r")
                    plt.savefig("temp/lm_lip_overlap.png")
                # check if all the lip landmarks are with in the mask bbox - [x1, y1, x2, y2]
                keep1 = (
                    np.all(lip_lm[:, 0] >= mask[0])
                    and np.all(lip_lm[:, 1] >= mask[1])
                    and np.all(lip_lm[:, 0] <= mask[2])
                    and np.all(lip_lm[:, 1] <= mask[3])
                )
                # check if all the landmarks are within the image
                keep2 = (
                    np.all(lm[:, 0] >= 0)
                    and np.all(lm[:, 0] <= 512)
                    and np.all(lm[:, 1] >= 0)
                    and np.all(lm[:, 1] <= 512)
                )
                keep = keep1 and keep2
                keeps.append(keep)
            # Atleast one false, we will remove the entire bins itself.
            if sum(keeps) == len(images):
                new_bins.append(bin)

                # print(f"Removing bin {n} for video {video} because it has landmarks outside the image")
        x[video]["clips"] = new_bins
        print(f"total bins: before: {len(bins)}, after: {len(x[video]['clips'])}")

    x = {k: v for k, v in x.items() if len(v["clips"]) > 2}
    total_bins_after = sum([len(v["clips"]) for v in x.values()])
    print("total bins, before: ", total_bins, "after: ", total_bins_after)
    save_json(json_path.parent / (json_path.name.rsplit(".")[0] + "_filtered.json"), x)


if __name__ == "__main__":
    typer.run(main)
