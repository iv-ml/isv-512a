import argparse
import os
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from lipsync.syncnet.model import SyncNetMulti


class SyncNetLightning(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        if self.opt.wav == 12:
            self.syncnet = SyncNetMulti(
                self.opt.img_length,
                self.opt.audio_length,
                1,
                self.opt.out_dim,
                backbone_name=self.opt.backbone_name,
                image_size=self.opt.mouth_region_size[self.opt.stage],
            )
        else:
            raise NotImplementedError("Its not implemented")
        self.train_loss = []
        self.val_loss = []

        self.optimzier = AdamW(self.syncnet.parameters(), lr=self.opt.lr)
        if self.opt.scheduler is not None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimzier, 10, 2)
        self.ss = torch.nn.CosineSimilarity()
        _LOSS = {"bce": torch.nn.BCELoss()}
        self.logloss = _LOSS[self.opt.loss_type]

    def forward(self, image, audio):
        return self.syncnet(image, audio)

    def cosine_loss(self, v: torch.Tensor, a: torch.Tensor, y: torch.Tensor):
        dd = self.cosine_score(v, a)
        return self.logloss(dd.unsqueeze(1), y)

    def cosine_score(self, v: torch.Tensor, a: torch.Tensor):
        if self.opt.normalize_embeddings:
            a = F.relu(a)
            v = F.relu(v)
            a = F.normalize(a, p=2, dim=1)
            v = F.normalize(v, p=2, dim=1)
        d = self.ss(a, v)
        return d.clip(1e-6, 1 - 1e-6)

    def training_step(self, batch, batch_idx):
        images, audio, y = batch
        image_features, audio_features = self(images, audio)
        if self.opt.loss == "cosine_loss":
            loss = self.cosine_loss(image_features, audio_features, y.float())
        else:
            raise NotImplementedError(f"not implemented loss type: {opt.loss}")

        pos_score = self.cosine_score(image_features[y[:, 0] == 1, :], audio_features[y[:, 0] == 1, :])

        neg_score = self.cosine_score(image_features[y[:, 0] == 0, :], audio_features[y[:, 0] == 0, :])

        self.log(
            "train_neg_score",
            neg_score.mean(),
            sync_dist=True,
            batch_size=images.shape[0],
        )
        self.log(
            "train_neg_score_std",
            neg_score.std(),
            sync_dist=True,
            batch_size=images.shape[0],
        )

        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            batch_size=images.shape[0],
            prog_bar=True,
        )

        self.log(
            "train_pos_score",
            pos_score.mean(),
            sync_dist=True,
            batch_size=images.shape[0],
            prog_bar=True,
        )

        self.log(
            "train_pos_score_std",
            pos_score.std(),
            sync_dist=True,
            batch_size=images.shape[0],
        )

        self.train_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, audio, y = batch
        image_features, audio_features = self(images, audio)
        if self.opt.loss == "cosine_loss":
            loss = self.cosine_loss(image_features, audio_features, y.float())
        else:
            raise NotImplementedError(f"not implemented loss type: {opt.loss}")

        pos_score = self.cosine_score(image_features[y[:, 0] == 1, :], audio_features[y[:, 0] == 1, :])

        neg_score = self.cosine_score(image_features[y[:, 0] == 0, :], audio_features[y[:, 0] == 0, :])

        self.log(
            "vbs",
            images.shape[0],
            sync_dist=True,
            batch_size=images.shape[0],
            prog_bar=True,
        )
        self.log(
            "val_loss",
            loss.mean(),
            sync_dist=True,
            batch_size=images.shape[0],
            prog_bar=True,
        )
        self.log(
            "val_pos_score",
            pos_score.mean(),
            sync_dist=True,
            batch_size=images.shape[0],
            prog_bar=True,
        )
        self.log(
            "val_neg_score",
            neg_score.mean(),
            sync_dist=True,
            batch_size=images.shape[0],
        )

        self.log(
            "val_pos_score_std",
            pos_score.std(),
            sync_dist=True,
            batch_size=images.shape[0],
        )
        self.log(
            "val_neg_score_std",
            neg_score.std(),
            sync_dist=True,
            batch_size=images.shape[0],
        )
        self.val_loss.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        loss = torch.stack(self.train_loss).mean()
        self.log("train_ep_loss", loss, sync_dist=True, prog_bar=True)
        self.train_loss.clear()

    def on_validation_epoch_end(self) -> None:
        score = torch.stack(self.val_loss).mean()
        self.log("val_ep_loss", score, sync_dist=True, prog_bar=True)
        self.val_loss.clear()

    def configure_optimizers(self):
        if self.opt.scheduler is not None:
            return [self.optimzier], [self.scheduler]
        else:
            return [self.optimzier]


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


# Usage example
if __name__ == "__main__":
    from lightning.pytorch.callbacks import (
        LearningRateMonitor,
        ModelCheckpoint,
        RichModelSummary,
        TQDMProgressBar,
    )
    from lightning.pytorch.loggers import WandbLogger
    from mmengine.config import Config

    from lipsync.syncnet.ds import load_concat_data

    # python scripts/train_conv.py --config_path="scripts/w2/exp2_hdtf_mead_iv1.py"
    args = get_args()
    opt = Config.fromfile(args.config_path)

    training_data_loader, val_data_loader = load_concat_data(opt)

    model = SyncNetLightning(opt)
    tqdm_callback = TQDMProgressBar(refresh_rate=4)
    lr_callback = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=30,
        mode="min",
        monitor="val_ep_loss",
        filename="{epoch}-{step}-{val_ep_loss:.3f}-{train_ep_loss:.3f}",
    )
    if not opt.no_wandb:
        wand_log = WandbLogger(name=opt.name, project=opt.project, save_dir=opt.save_dir)
    else:
        wand_log = None
    summary = RichModelSummary(max_depth=3)

    trainer = L.Trainer(
        accelerator=opt.accelarator,
        devices=opt.devices,
        max_epochs=opt.epochs,
        num_nodes=int(os.environ.get("NUM_NODES", 1)),
        strategy="ddp" if opt.devices > 1 else "auto",
        num_sanity_val_steps=2,
        sync_batchnorm=True,
        log_every_n_steps=5,
        # gradient_clip_val=1,
        callbacks=[tqdm_callback, lr_callback, checkpoint_callback],
        logger=wand_log,
    )

    trainer.fit(model, train_dataloaders=training_data_loader, val_dataloaders=val_data_loader)