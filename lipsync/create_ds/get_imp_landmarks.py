LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LIP_IDS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375]

if __name__ == "__main__":
    import fastcore.all as fc
    import numpy as np
    from tqdm import tqdm

    from lipsync.utils import load_json, save_json

    ds_name = "iv_recording_v2"
    landmarks_dir = fc.Path(f"/data/mp_lms/{ds_name}")
    bins_path = fc.Path(f"/data/prakash_lipsync/video_hallo_512/bins/{ds_name}")
    x = load_json(f"/data/prakash_lipsync/v1/{ds_name}/val_hallo.json")
    videos = fc.L(x.keys())

    for video in tqdm(videos):
        bins = x[video]["clips"]
        # load its landmarks
        landmarks = np.load(landmarks_dir / (video + ".npy"))
        bbox = np.load(bins_path / (video + ".npy"))
        # frame_number_to_bin_number = {b[1]:b[0] for n in bbox for b in n}
        for n, bin in enumerate(bins):
            images = bin["images"]
            lms = []
            for m, img in enumerate(images):
                frame_number = int(img.rsplit(".")[0])
                # bbox is of shape (n_bins, 5, 6) - bin_number, frame_number, x1, y1, x2, y2, we need to get the index of the frame_number
                bin_number = np.where(bbox[:, :, 1] == frame_number)[0]
                _, frame_number_2, x1, y1, x2, y2 = bbox[bin_number][0][m]
                lm = landmarks[frame_number_2, :, :2].copy()
                if frame_number != frame_number_2:
                    print(frame_number, frame_number_2)
                    breakpoint()
                lm[:, 0] = (lm[:, 0] - x1) / (x2 - x1) * 512
                lm[:, 1] = (lm[:, 1] - y1) / (y2 - y1) * 512

                eye_center = np.mean(lm[LEFT_EYE + RIGHT_EYE, :], 0)
                lip_center = np.mean(lm[LIP_IDS, :], 0)
                lms.append(
                    {
                        "eye_center": [int(i) for i in np.round(eye_center).tolist()],
                        "lip_center": [int(i) for i in np.round(lip_center).tolist()],
                    }
                )
            x[video]["clips"][n]["lms"] = lms
    save_json(f"/data/prakash_lipsync/v1/{ds_name}/val_hallo_512_lm.json", x)
