# Clip is trained on Sync + PerceptionLoss + GAN Loss.

import argparse
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torchvision.utils import make_grid

from lipsync.dinet.discriminator import EnhancedDiscriminator
from lipsync.dinet.losses import GANLoss, PerceptionLoss
from lipsync.dinet_lp.model import DINetSPADE
from lipsync.syncnet.infer import SyncNetPerceptionMulti
from lipsync.utils import check_gradient_magnitudes


def mask2bbox(mask):
    rows = torch.any(mask, axis=1)
    cols = torch.any(mask, axis=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax


def get_scheduler(optimizer, niter, niter_decay, lr_policy="lambda", lr_decay_iters=50):
    """
    scheduler in training stage
    """
    from torch.optim import lr_scheduler

    if lr_policy == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - niter) / float(niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        raise ValueError(f"Unsupported lr policy: {lr_policy}")
    return scheduler


def vis_images_on_wandb(source_image_data, source_image_mask, predictions, batch_size):
    source_image_data = torch.cat(torch.split(source_image_data, batch_size, dim=0), 1)
    source_image_mask = torch.cat(torch.split(source_image_mask, batch_size, dim=0), 1)
    predictions = torch.cat(torch.split(predictions, batch_size, dim=0), 1)
    h_, w_ = source_image_data[0].shape[1:]
    count = source_image_data.shape[1] // 3
    images = torch.cat(
        [
            source_image_data[0].reshape(count, 3, h_, w_),
            source_image_mask[0].reshape(count, 3, h_, w_),
            predictions[0].reshape(count, 3, h_, w_),
        ],
        dim=1,
    )
    grid = make_grid(images.reshape(source_image_data.shape[1], 3, h_, w_).clip(0, 1), nrow=3)
    return grid


class DINetClipLightningModule(L.LightningModule):
    def __init__(self, opt):
        super(DINetClipLightningModule, self).__init__()
        self.automatic_optimization = False
        self.opt = opt
        self.net_g = DINetSPADE(
            opt.source_channel,
            opt.ref_channel,
            audio_seq_len=opt.audio_seq_len,
            upscale=opt.upscale,
        )
        self.net_dI = EnhancedDiscriminator(
            opt.source_channel, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features
        )
        self.lip_net_dI = EnhancedDiscriminator(
            opt.source_channel, opt.LipD_block_expansion, opt.LipD_num_blocks, opt.LipD_max_features
        )
        self.net_lipsync = SyncNetPerceptionMulti(opt.pretrained_syncnet_path)

        self.perception_loss = PerceptionLoss()
        self.gan_loss = GANLoss()

        self.train_sync_loss = []
        self.val_sync_loss = []
        self.train_loss = []
        self.val_loss = []

        if opt.pretrained_frame_DINet_path is not None:
            print("Loading Genertor model")
            ckpt = torch.load(opt.pretrained_frame_DINet_path)
            ckpt_netg = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net_g.")}
            actual_state_dict = self.net_g.state_dict()
            new = {}
            for name, param in actual_state_dict.items():
                if name in ckpt_netg.keys():
                    if param.shape == ckpt_netg[name].shape:
                        print(f"Transfering weights: {name}-{param.shape}")
                        new[name] = ckpt_netg[name]
                    else:
                        print(f"Unable to transfer weights: {name}-{param.shape}")
                        new[name] = actual_state_dict[name]
                else:
                    print(f"Unable to transfer weights: {name}-{param.shape} not available")
                    new[name] = actual_state_dict[name]
            self.net_g.load_state_dict(new)

        # self.save_hyperparameters(self.opt)

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        ckpt_netg = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net_g.")}
        self.net_g.load_state_dict(ckpt_netg)

    def forward(self, source_image, reference_clip_data, deepspeech_feature, source_mask):
        return self.net_g(source_image, reference_clip_data, deepspeech_feature, source_mask)

    def on_train_epoch_end(self):
        if self.opt.scheduler not in ["cosine_warmrestarts", "cosine_lr_with_warmup"]:
            self.net_g_scheduler.step()
            self.net_dI_scheduler.step()
            self.lip_net_dI_scheduler.step()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Step the learning rate schedulers
        if self.opt.scheduler in ["cosine_warmrestarts", "cosine_lr_with_warmup"]:
            self.net_g_scheduler.step()
            self.net_dI_scheduler.step()
            self.lip_net_dI_scheduler.step()

        sync_loss = torch.stack(self.train_sync_loss).mean()
        self.log("train_ep_sync_loss", sync_loss, sync_dist=True, prog_bar=True)
        self.train_sync_loss.clear()

        loss = torch.stack(self.train_loss).mean()
        self.log("train_ep_loss", loss, sync_dist=True, prog_bar=True)
        self.train_loss.clear()

    def training_step(self, batch, batch_idx):
        opt_g, opt_d, opt_lip_d = self.optimizers()
        (source_clip_v, source_mask_v, reference_clip, deep_speech_clip, deep_speech_full, _) = batch

        gt_source_clip = torch.cat(torch.split(source_clip_v, 1, dim=1), 0).squeeze(1)
        gt_source_mask = torch.cat(torch.split(source_mask_v, 1, dim=1), 0).squeeze(1)
        gt_reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1)
        deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1)

        if self.opt.upscale > 1:
            source_clip = F.interpolate(gt_source_clip, scale_factor=1 / self.opt.upscale, mode="bilinear")
            source_mask = F.interpolate(gt_source_mask.float(), scale_factor=1 / self.opt.upscale, mode="nearest")
            source_mask = (source_mask > 0.5).float()
            reference_clip = F.interpolate(gt_reference_clip, scale_factor=1 / self.opt.upscale, mode="bilinear")
        else:
            source_clip = gt_source_clip
            reference_clip = gt_reference_clip
            source_mask = gt_source_mask.float()

        fake_out = self(source_clip, reference_clip, deep_speech_clip, source_mask=source_mask)
        fake_out_resized = F.interpolate(fake_out, scale_factor=1 / self.opt.upscale, mode="bilinear")
        lip_union_mask = (gt_source_mask == 0).sum(axis=(0, 1)) > 0
        xmin, ymin, xmax, ymax = mask2bbox(lip_union_mask)
        lip_output = fake_out[:, :, ymin:ymax, xmin:xmax]
        gt_lip_portion = gt_source_clip[:, :, ymin:ymax, xmin:xmax]

        # lip_output = torch.where(gt_source_mask == 0, fake_out, torch.zeros_like(fake_out))
        # gt_lip_portion = torch.where(gt_source_mask == 0, gt_source_clip, torch.zeros_like(gt_source_clip))

        # face discriminator
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        loss_d_face, _ = self.discriminator(fake_out, gt_source_clip)
        self.manual_backward(loss_d_face)

        if self.opt.clip_gradients:
            pre_clip_norm = check_gradient_magnitudes(self.net_dI)
            self.clip_gradients(self.net_dI, self.opt.disc_clip_value)
            post_clip_norm = check_gradient_magnitudes(self.net_dI)
            self.log(name="img_disc_post", value=post_clip_norm, batch_size=source_clip.shape[0], sync_dist=True)
            self.log(name="img_disc_pre", value=pre_clip_norm, batch_size=source_clip.shape[0], sync_dist=True)

        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # lip discriminator
        self.toggle_optimizer(opt_lip_d)
        opt_lip_d.zero_grad()
        loss_d_lip, _ = self.lip_discriminator(lip_output, gt_lip_portion)
        self.manual_backward(loss_d_lip)

        if self.opt.clip_gradients:
            lip_pre_clip_norm = check_gradient_magnitudes(self.lip_net_dI)
            self.clip_gradients(self.lip_net_dI, self.opt.disc_clip_value)
            lip_post_clip_norm = check_gradient_magnitudes(self.lip_net_dI)
            self.log(
                name="lip_img_disc_post", value=lip_post_clip_norm, batch_size=source_clip.shape[0], sync_dist=True
            )
            self.log(name="lip_img_disc_pre", value=lip_pre_clip_norm, batch_size=source_clip.shape[0], sync_dist=True)

        opt_lip_d.step()
        self.untoggle_optimizer(opt_lip_d)

        loss_d = (loss_d_face + loss_d_lip) / 3

        # genertor
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()

        _, pred_fake_dI = self.net_dI(fake_out)
        _, pred_fake_lip_dI = self.lip_net_dI(lip_output)
        loss = {}
        loss["perception_loss"] = self.perception_loss(fake_out, gt_source_clip) * self.opt.lambda_perception
        loss["l1_loss"] = F.l1_loss(fake_out, gt_source_clip) * self.opt.lambda_perception * 5
        loss["perception_lip_loss"] = self.perception_loss(lip_output, gt_lip_portion) * self.opt.lambda_perception
        loss["l1_lip_loss"] = F.l1_loss(lip_output, gt_lip_portion) * self.opt.lambda_perception * 5
        loss["sync_loss"] = (
            self.process_sync(fake_out_resized, deep_speech_full.type_as(fake_out)) * self.opt.lambda_syncnet_perception
        )
        loss["gen_adv_loss"] = self.gan_loss(pred_fake_dI, True)
        loss["lip_gen_adv_loss"] = self.gan_loss(pred_fake_lip_dI, True)
        loss["total_gen_loss"] = (
            loss["perception_loss"]
            + loss["l1_loss"]
            + loss["perception_lip_loss"]
            + loss["l1_lip_loss"]
            + loss["gen_adv_loss"]
            + loss["lip_gen_adv_loss"]
        )

        self.manual_backward(loss["total_gen_loss"])
        if self.opt.clip_gradients:
            pre_clip_norm = check_gradient_magnitudes(self.net_g)
            self.clip_gradients(self.net_g, self.opt.gen_clip_value)
            post_clip_norm = check_gradient_magnitudes(self.net_g)
            self.log(
                name="gn_post",
                value=post_clip_norm,
                batch_size=source_clip.shape[0],
                sync_dist=True,
            )
            self.log(
                name="gn_pre",
                value=pre_clip_norm,
                batch_size=source_clip.shape[0],
                sync_dist=True,
            )
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        loss["disc_loss"] = loss_d_face
        loss["lip_disc_loss"] = loss_d_lip
        loss["total_disc_loss"] = loss_d

        for k, v in loss.items():
            self.log(
                name=f"train_{k}",
                value=v,
                prog_bar=True,
                batch_size=self.opt.batch_size,
                sync_dist=True,
            )

        self.train_sync_loss.append(loss["sync_loss"])
        self.train_loss.append(loss["total_gen_loss"])

        if batch_idx == 0 and self.trainer.is_global_zero:
            gt_source_clip_mask = gt_source_clip.clone().detach() * gt_source_mask
            grid = vis_images_on_wandb(gt_source_clip, gt_source_clip_mask, fake_out, self.opt.batch_size)
            wgrid = wandb.Image(
                grid,
                caption=f"loss_g: {loss['total_gen_loss'].item():.4f}, loss_di: {loss['total_disc_loss'].item():.4f}, epoch: {self.current_epoch}",
            )
            wandb.log({"images_train": [wgrid]})

    def validation_step(self, batch, batch_idx):
        (source_clip_v, source_mask_v, reference_clip, deep_speech_clip, deep_speech_full, _) = batch

        gt_source_clip = torch.cat(torch.split(source_clip_v, 1, dim=1), 0).squeeze(1)
        gt_source_mask = torch.cat(torch.split(source_mask_v, 1, dim=1), 0).squeeze(1)
        gt_reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1)
        deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1)

        if self.opt.upscale > 1:
            source_clip = F.interpolate(gt_source_clip, scale_factor=1 / self.opt.upscale, mode="bilinear")
            source_mask = F.interpolate(gt_source_mask.float(), scale_factor=1 / self.opt.upscale, mode="nearest")
            source_mask = (source_mask > 0.5).float()
            reference_clip = F.interpolate(gt_reference_clip, scale_factor=1 / self.opt.upscale, mode="bilinear")
        else:
            source_clip = gt_source_clip
            reference_clip = gt_reference_clip
            source_mask = gt_source_mask.float()

        fake_out = self(source_clip, reference_clip, deep_speech_clip, source_mask=source_mask)
        fake_out_resized = F.interpolate(fake_out, scale_factor=1 / self.opt.upscale, mode="bilinear")
        # should we draw a bbox across all the gt_source_mask and calculate the loss?
        # get union mask
        lip_union_mask = (gt_source_mask == 0).sum(axis=(0, 1)) > 0
        xmin, ymin, xmax, ymax = mask2bbox(lip_union_mask)
        lip_output = fake_out[:, :, ymin:ymax, xmin:xmax]
        gt_lip_portion = gt_source_clip[:, :, ymin:ymax, xmin:xmax]

        # import torchvision
        # torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(lip_output, nrow=5)).save("temp/lip_output.png")
        # torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(gt_lip_portion, nrow=5)).save("temp/gt_lip_output.png")
        # torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(fake_out, nrow=5)).save("temp/gt_full_output.png")

        # lip_output = torch.where(gt_source_mask == 0, fake_out, torch.zeros_like(fake_out))
        # gt_lip_portion = torch.where(gt_source_mask == 0, gt_source_clip, torch.zeros_like(gt_source_clip))
        _, pred_fake_dI = self.net_dI(fake_out)
        _, pred_fake_lip_dI = self.lip_net_dI(lip_output)
        loss = {}
        # Generator loss
        ## insight face loss  TODO
        ## lip distance l1 loss TODO
        loss["perception_loss"] = self.perception_loss(fake_out, gt_source_clip) * self.opt.lambda_perception
        loss["l1_loss"] = F.l1_loss(fake_out, gt_source_clip) * self.opt.lambda_perception * 5
        loss["perception_lip_loss"] = self.perception_loss(lip_output, gt_lip_portion) * self.opt.lambda_perception
        loss["l1_lip_loss"] = F.l1_loss(lip_output, gt_lip_portion) * self.opt.lambda_perception * 5
        loss["sync_loss"] = (
            self.process_sync(fake_out_resized, deep_speech_full.type_as(fake_out)) * self.opt.lambda_syncnet_perception
        )
        loss["gen_adv_loss"] = self.gan_loss(pred_fake_dI, True)
        loss["lip_gen_adv_loss"] = self.gan_loss(pred_fake_lip_dI, True)
        loss["total_gen_loss"] = (
            loss["perception_loss"]
            + loss["l1_loss"]
            + loss["perception_lip_loss"]
            + loss["l1_lip_loss"]
            + loss["gen_adv_loss"]
            + loss["lip_gen_adv_loss"]
        )

        # Discriminator loss
        loss["disc_loss"], _ = self.discriminator(fake_out, gt_source_clip)
        loss["lip_disc_loss"], _ = self.lip_discriminator(lip_output, gt_lip_portion)
        loss["total_disc_loss"] = loss["disc_loss"] + loss["lip_disc_loss"]

        for k, v in loss.items():
            self.log(
                name=f"val_{k}",
                value=v,
                prog_bar=True,
                batch_size=self.opt.batch_size,
                sync_dist=True,
            )

        if batch_idx == 0 and self.trainer.is_global_zero:
            gt_source_clip_mask = gt_source_clip.clone().detach() * gt_source_mask
            grid = vis_images_on_wandb(gt_source_clip, gt_source_clip_mask, fake_out, self.opt.batch_size)
            wgrid = wandb.Image(
                grid,
                caption=f"loss_g: {loss['total_gen_loss']:.4f}, loss_di: {loss['total_disc_loss']:.4f}, epoch: {self.current_epoch}",
            )
            wandb.log({"images": [wgrid]})

        self.val_loss.append(loss["total_gen_loss"].detach())
        self.val_sync_loss.append(loss["sync_loss"])

    def on_validation_epoch_end(self) -> None:
        sync_loss = torch.stack(self.val_sync_loss).mean()
        self.log("val_ep_sync_loss", sync_loss, sync_dist=True, prog_bar=True)
        self.val_sync_loss.clear()

        loss = torch.stack(self.val_loss).mean()
        self.log("val_ep_loss", loss, sync_dist=True, prog_bar=True)
        self.val_loss.clear()

    def process_sync(self, fake_out, deep_speech_full):
        """
        for mutli it should of shape (batch_size, N_images, N_channels, h, w) (16, 5, 3, 128, 128)
        """

        fake_out_mouth = fake_out[:, :, 112 : 240 - 48, 64:192]
        fake_out_mouth = torch.cat(
            [i.unsqueeze(1) for i in torch.split(fake_out_mouth, self.opt.batch_size, dim=0)],
            1,
        )
        sync_score = self.net_lipsync(fake_out_mouth, deep_speech_full).clip(1e-6, 1.0 - 1.0e-6)
        loss_sync = -1 * torch.mean(torch.log(sync_score))
        return loss_sync

    def discriminator(self, fake_out, source_image_data):
        _, pred_fake_dI = self.net_dI(fake_out.detach())
        loss_dI_fake = self.gan_loss(pred_fake_dI, False)

        real_features, pred_real_dI = self.net_dI(source_image_data)
        loss_dI_real = self.gan_loss(pred_real_dI, True)
        loss_dI = (loss_dI_fake + loss_dI_real) * 0.5
        return loss_dI, real_features

    def lip_discriminator(self, fake_out, source_image_data):
        _, pred_fake_dI = self.lip_net_dI(fake_out.detach())
        loss_dI_fake = self.gan_loss(pred_fake_dI, False)

        real_features, pred_real_dI = self.lip_net_dI(source_image_data)
        loss_dI_real = self.gan_loss(pred_real_dI, True)
        loss_dI = (loss_dI_fake + loss_dI_real) * 0.5
        return loss_dI, real_features

    def configure_optimizers(self):
        opt_g = optim.AdamW(self.net_g.parameters(), lr=opt.lr_g, fused=True)
        opt_di = optim.AdamW(self.net_dI.parameters(), lr=opt.lr_dI, fused=True)
        opt_lip_di = optim.AdamW(self.lip_net_dI.parameters(), lr=opt.lr_lip_dI, fused=True)

        self.net_g_scheduler = get_scheduler(opt_g, opt.non_decay, opt.decay, lr_policy=self.opt.scheduler)
        self.net_dI_scheduler = get_scheduler(opt_di, opt.non_decay, opt.decay, lr_policy=self.opt.scheduler)
        self.lip_net_dI_scheduler = get_scheduler(opt_lip_di, opt.non_decay, opt.decay, lr_policy=self.opt.scheduler)
        return [opt_g, opt_di, opt_lip_di], [
            self.net_g_scheduler,
            self.net_dI_scheduler,
            self.lip_net_dI_scheduler,
        ]

    def clip_gradients(self, model, clip_value):
        for param in model.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, clip_value)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


def get_node_rank():
    import os

    if "SLURM_NODEID" in os.environ:
        return int(os.environ["SLURM_NODEID"])
    if "NODE_RANK" in os.environ:
        return int(os.environ["NODE_RANK"])
    return 0  # default to 0 if not in a multi-node environment


if __name__ == "__main__":
    from lightning.pytorch.callbacks import (
        LearningRateMonitor,
        ModelCheckpoint,
        TQDMProgressBar,
    )
    from lightning.pytorch.loggers import WandbLogger
    from loguru import logger
    from mmengine.config import Config

    torch.set_float32_matmul_precision("high")

    # python scripts/train_clip.py --config_path="scripts/config/exp1_64_clip.py"
    args = get_args()
    opt = Config.fromfile(args.config_path)

    if "video_folder_lp" in opt.data[0].keys():
        from lipsync.dinet_lp.ds import load_concat_data
    else:
        from lipsync.dinet_lp.ds_static import load_concat_data

    training_data_loader, val_data_loader = load_concat_data(opt)

    model = DINetClipLightningModule(opt)

    tqdm_callback = TQDMProgressBar(refresh_rate=4)
    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=100,
        monitor="val_ep_loss",
        filename="{epoch}-{step}-{val_ep_loss:.3f}-{train_ep_loss:.3f}-{val_ep_sync_loss:.3f}-{train_ep_sync_loss:.3f}",
    )

    wand_log = None
    node_rank = 0
    if not opt.no_wandb:
        if torch.distributed.is_initialized():
            logger.info("using wandb logger with distributed training")
            node_rank = get_node_rank()
            if (node_rank == 0) & (model.global_rank == 0):
                wand_log = WandbLogger(
                    project=opt.clip_project,
                    name=opt.clip_name,
                    save_dir=opt.save_dir,
                )
                # wand_log.watch(model.net_g, log="all", log_freq=100)
        else:
            wand_log = WandbLogger(
                project=opt.clip_project,
                name=opt.clip_name,
                save_dir=opt.save_dir,
            )
            # wand_log.watch(model.net_g, log="all", log_freq=100)

    trainer = L.Trainer(
        num_nodes=opt.nodes,
        accelerator=opt.accelarator,
        devices=opt.devices,
        max_epochs=opt.non_decay + opt.decay + 1,
        strategy="ddp_find_unused_parameters_true" if opt.devices > 1 else "auto",
        num_sanity_val_steps=2,
        sync_batchnorm=True,
        log_every_n_steps=5,
        check_val_every_n_epoch=opt.check_val_every_n_epoch,
        callbacks=[tqdm_callback, lr_callback, checkpoint_callback],
        logger=wand_log,
    )
    trainer.fit(
        model,
        train_dataloaders=training_data_loader,
        val_dataloaders=val_data_loader,
    )
