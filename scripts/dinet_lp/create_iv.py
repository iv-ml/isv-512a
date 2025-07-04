# For iv2 we need to keep only the clips for which we have lp outputs too


if __name__ == "__main__":
    import fastcore.all as fc
    from tqdm import tqdm

    from lipsync.utils import load_json, save_json

    x = fc.L(fc.Path("/data/prakash_lipsync/video_hallo_512/iv_recording_v2/").glob("*"))
    lp_root = fc.Path("/data/prakash_lipsync/video_hallo_512_lp/iv_recording_v2/")
    print(len(x))

    names = []
    for f in tqdm(x):
        imgs = fc.L(f.glob("*/*.png"))
        lp_imgs = fc.L((lp_root / f.name).glob("*/*.png"))
        if len(lp_imgs) == len(imgs):
            names.append(f.name)

    print("loading json")
    x1 = load_json(fc.Path("/data/prakash_lipsync/v1/iv_recording_v2/train_hallo_512_lm.json"))
    x2 = load_json(fc.Path("/data/prakash_lipsync/v1/iv_recording_v2/val_hallo_512_lm.json"))
    print("update dicts")
    x1 = {k: v for k, v in x1.items() if k in names}
    x2 = {k: v for k, v in x2.items() if k in names}
    print(len(x1), len(x2))
    save_json(fc.Path("/data/prakash_lipsync/v1/iv_recording_v2/train_hallo_512_lm_lp.json"), x1)
    save_json(fc.Path("/data/prakash_lipsync/v1/iv_recording_v2/val_hallo_512_lm_lp.json"), x2)
