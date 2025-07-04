from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import typer
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from mmengine.config import Config

from lipsync.syncnet.ds import SyncNetDataset, CombinedDataset
from lipsync.syncnet.infer import SyncNetPerceptionMulti


def get_dataloader(ds_name: str, cfg: dict):
    ds_cfg = [ds_cfg for ds_cfg in cfg["data"] if ds_cfg["ds_name"] == ds_name][0]
    ds_val = SyncNetDataset(
        data_root=ds_cfg["data_root"],
        ds_name=ds_name,
        audio_features_name=ds_cfg["audio_features_name"],
        json_loc=ds_cfg["val_json"],
        mouth_region_size=cfg["mouth_region_size"][cfg["stage"]],
        train_val="val",
    )
    ds_train = SyncNetDataset(
        data_root=ds_cfg["data_root"],
        ds_name=ds_name,
        audio_features_name=ds_cfg["audio_features_name"],
        json_loc=ds_cfg["train_json"],
        mouth_region_size=cfg["mouth_region_size"][cfg["stage"]],
        train_val="val",
    )
    ds = CombinedDataset([ds_train, ds_val])
    return DataLoader(ds, batch_size=2048, shuffle=False, num_workers=64)


def main(
    ckpt: Path = typer.Argument(..., help="Path to the checkpoint"),
    ds_name: str = typer.Argument(..., help="Name of the dataset"),
    cfg: Path = typer.Option(..., help="Path to the config file"),
    dst: Path = typer.Option("results", help="Path to the output csv"),
):
    model = SyncNetPerceptionMulti(ckpt)
    cfg = Config.fromfile(cfg)
    dl = get_dataloader(ds_name, cfg)
    scores = []
    for batch in tqdm(dl, total=len(dl), desc=f"Inferring {ds_name}"):
        source_lip, audio_feats_batch, silences, bin_id = batch
        with torch.no_grad():
            sync_score = model(source_lip.cuda(), audio_feats_batch.cuda())
            sync_score = sync_score.clip(1e-6, 1.0 - 1.0e-6)
        score = [(id_, int(silence), score.item()) for id_, silence, score in zip(bin_id, silences, sync_score.cpu())]
        scores.extend(score)
    df = pd.DataFrame(scores, columns=["clip_id", "silence", "score"])
    df["bin"] = df["clip_id"].str.split("_").str[-1].astype(int)
    df["clip_id"] = df["clip_id"].str.split("_").str[:-1].str.join("_")
    df.sort_values(by=["clip_id", "bin"], ascending=[True, True], inplace=True)
    csv_path = dst / f"sync_scores_{ds_name}.csv"
    df.to_csv(csv_path, index=False)
    print(df.head(10))
    print(df.tail(10))
    logger.info(f"Saved to {csv_path}")


if __name__ == "__main__":
    typer.run(main)