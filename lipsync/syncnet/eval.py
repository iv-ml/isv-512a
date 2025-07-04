import torch
import torch.nn.functional as F
import torchvision
from safetensors.torch import load_file


def load_imgs(imgs_loc, upscale=2, mask_dim=[64, 112, 192, 240]):
    imgs = [torchvision.io.read_image(i) for i in imgs_loc]
    imgs = torch.stack(imgs)
    imgs = F.interpolate(imgs, scale_factor=1 / upscale, mode="bilinear")
    imgs = imgs[:, :, mask_dim[1] : mask_dim[3], mask_dim[0] : mask_dim[2]] / 255.0
    return imgs


if __name__ == "__main__":
    import fastcore.all as fc
    import numpy as np
    import pandas as pd

    from lipsync.syncnet.infer import SyncNetPerceptionMulti

    folders = fc.L(fc.Path("/data/prakash_lipsync/video_hallo_512/th_1kh_512").glob("*"))
    folder = folders[np.random.randint(0, len(folders))]

    model = SyncNetPerceptionMulti("weights/v9_syncnet/e134_256_e134-s17415-v194-t192-hdtf_iv_th1kh512.ckpt")

    scores = []
    import random

    random.seed(42)
    random.shuffle(folders)
    for f in folders[:100]:
        bins = fc.L(f.glob("*"))
        audio_path = fc.Path("/data/prakash_lipsync/audio/th_1kh_512") / (f.name + ".safetensors")
        if not audio_path.exists():
            continue
        audio = load_file(audio_path)["audio_embedding"]
        for b in bins:
            imgs_loc = fc.L(b.glob("*.png"))
            imgs_loc.sort()
            if len(imgs_loc) < 5:
                continue
            imgs = load_imgs(imgs_loc)
            start_audio, end_audio = int(imgs_loc[0].stem), int(imgs_loc[-1].stem)
            audio_bin = audio[start_audio - 2 : end_audio + 2 + 1]
            score = model(imgs[None].cuda(), audio_bin[None].cuda())
            scores.append([f.name, int(b.name), round(score.detach().cpu().numpy().tolist()[0], 3)])
            print(scores[-1])

        print("storing scores")
        df = pd.DataFrame(scores, columns=["video", "bin", "score"])
        df.to_csv("temp/scores_th_1kh_512_v9.csv", index=False)
        print(df["score"].mean())
