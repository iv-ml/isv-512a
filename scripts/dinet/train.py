# Clip is trained on Sync + PerceptionLoss + GAN Loss.

import argparse
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from einops import rearrange
from torchvision.utils import make_grid

import wandb
from lipsync.dinet.discriminator import EnhancedDiscriminator
from lipsync.dinet.losses import GANLoss, PerceptionLoss
from lipsync.dinet.model import DINetSPADE
from lipsync.syncnet.infer import SyncNetPerceptionMulti
from lipsync.utils import check_gradient_magnitudes

torch.multiprocessing.set_sharing_strategy("file_system")


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


def create_clip_image_grid(
    src,  # (b, c, t, h, w)
    dest,  # (b, c, t, h, w)
    dest_512,  # (b, c, t, h, w)
    references,  # (b, 15, c, h, w)
    path="/tmp/grid.gif",
    duration=0.5,
    loop=0,
    optimize=True,
):
    # resiz src, dest and dest_512 to 256x256
    # src = F.interpolate(src, size=(256, 256), mode="bilinear")
    # dest = F.interpolate(dest, size=(256, 256), mode="bilinear")
    # dest_512 = F.interpolate(dest_512, size=(256, 256), mode="bilinear")
    tensor = torch.cat([src, dest, dest_512], dim=4)
    references = rearrange(references, "b (f c) h w -> b c h (f w)", f=5)
    ref_h, ref_w = references.shape[-2:]

    ref_w_new = 3 * 512  # 3 = src + dest + dest_512
    ref_h_new = ref_h * ref_w_new // ref_w
    references = F.interpolate(references, size=(ref_h_new, ref_w_new), mode="bilinear")
    references = rearrange(references, "(t b) c h w -> b c t h w", t=5)
    # gt_reference_clip = rearrange(gt_reference_clip, "(t b) c h w -> b c t h w", t=5)
    # tensor = tensor[:, [2, 1, 0], :, :, :]
    tensor = torch.cat([tensor, references], dim=3)
    grid = [make_grid(t[:16].clip(0, 1), nrow=1) for t in tensor.unbind(dim=2)]
    images = map(torchvision.transforms.ToPILImage(), grid)
    first_img, *rest_imgs = images
    rest_img_copy = rest_imgs.copy()
    rest_img_copy.reverse()
    rest_imgs = rest_imgs + rest_img_copy
    first_img.save(
        path,
        save_all=True,
        append_images=rest_imgs,
        duration=duration,
        loop=loop,
        optimize=optimize,
    )


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
            reference_frames_process=opt.reference_frames_process,
            seg_face=opt.seg_face,
            use_attention=opt.use_attention,
        )
        self.mask_dim = [64, 112, 192, 240] if opt.image_size == 256 else [128, 224, 384, 480]
        self.net_dI = EnhancedDiscriminator(
            opt.source_channel, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features
        )
        self.lip_net_dI = EnhancedDiscriminator(
            opt.source_channel, opt.LipD_block_expansion, opt.LipD_num_blocks, opt.LipD_max_features
        )
        self.net_lipsync = SyncNetPerceptionMulti(opt.pretrained_syncnet_path)

        self.perception_loss = PerceptionLoss()
        self.gan_loss = GANLoss()
        if self.opt.identity_loss:
            from lipsync.identity_loss.loss import IdentityLoss

            self.identity_loss = IdentityLoss(opt.identity_loss_path, "cuda")
        else:
            self.identity_loss = None

        self.train_sync_loss = []
        self.val_sync_loss = []
        self.train_loss = []
        self.val_loss = []
        self.sync_dist = True
        # Add these to your existing initialization
        # self.train_discriminator = True
        self.train_lip_discriminator = False
        # self.disc_loss_threshold = 0.05
        # self.lip_disc_loss_threshold = 0.05
        # self.disc_losses = []  # Store losses on CPU during epoch
        # self.lip_disc_losses = []

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
            self.net_g.load_state_dict(new, strict=False)
            # print(f"missing_keys: {tmp_keys['missing_keys']}")
            # print(f"unexpected_keys: {tmp_keys['unexpected_keys']}")

        self.save_hyperparameters(self.opt.to_dict())

    def on_train_start(self):
        if self.opt.seg_face:
            self.net_g.segface.model.to(self.device)

    def on_validation_start(self):
        if self.opt.seg_face:
            self.net_g.segface.model.to(self.device)

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        ckpt_netg = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net_g.")}
        self.net_g.load_state_dict(ckpt_netg)

    def forward(self, source_image, reference_clip_data, deepspeech_feature, mask_dim):
        return self.net_g(source_image, reference_clip_data, deepspeech_feature, mask_dim)

    def on_train_epoch_end(self):
        if self.opt.scheduler not in ["cosine_warmrestarts", "cosine_lr_with_warmup"]:
            self.net_g_scheduler.step()
            self.net_dI_scheduler.step()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Step the learning rate schedulers
        if self.opt.scheduler in ["cosine_warmrestarts", "cosine_lr_with_warmup"]:
            self.net_g_scheduler.step()
            self.net_dI_scheduler.step()
        sync_loss = torch.stack(self.train_sync_loss).mean()
        self.log("train_ep_sync_loss", sync_loss, sync_dist=True, prog_bar=True)
        self.train_sync_loss.clear()

        loss = torch.stack(self.train_loss).mean()
        self.log("train_ep_loss", loss, sync_dist=True, prog_bar=True)
        self.train_loss.clear()

    def dict2string(self, dict):
        return "```\n" + pd.DataFrame([[k, v] for k, v in dict.items()]).to_markdown(index=False) + "\n```"

    def training_step(self, batch, batch_idx):
        opt_g, opt_d, opt_lip_d = self.optimizers()
        (source_clip_v, reference_clip, deep_speech_clip, deep_speech_full, _) = batch

        gt_source_clip = torch.cat(torch.split(source_clip_v, 1, dim=1), 0).squeeze(1)
        gt_reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1)
        deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1)
        # add noise to first reference frame
        noise_weight = 0.2 / np.log(np.e + self.current_epoch**4)
        gt_reference_clip[:, :3, :, :] += noise_weight * torch.randn_like(gt_reference_clip[:, :3, :, :])
        gt_reference_clip[:, :3, :, :] = torch.clamp(gt_reference_clip[:, :3, :, :], 0, 1)
        if self.opt.upscale > 1:
            source_clip = F.interpolate(gt_source_clip, scale_factor=1 / self.opt.upscale, mode="bilinear")
            # reference_clip = F.interpolate(gt_reference_clip, scale_factor=1 / self.opt.upscale, mode="bilinear")
        else:
            source_clip = gt_source_clip
        reference_clip = gt_reference_clip
        fake_out, reference_clip = self(
            source_clip,
            reference_clip,
            deep_speech_clip,
            mask_dim=self.mask_dim,
        )
        gt_source_clip = source_clip.clone()
        fake_out_resized = F.interpolate(fake_out, scale_factor=1 / self.opt.upscale, mode="bilinear")
        self.train_discriminator = True
        self.train_lip_discriminator = True
        # face discriminator
        loss_d_face, _ = self.discriminator(fake_out, gt_source_clip)
        if self.train_discriminator:
            self.toggle_optimizer(opt_d)
            opt_d.zero_grad()
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
        loss_d_lip, _ = self.lip_discriminator(fake_out, gt_source_clip)
        if self.train_lip_discriminator:
            self.toggle_optimizer(opt_lip_d)
            opt_lip_d.zero_grad()
            self.manual_backward(loss_d_lip)

            if self.opt.clip_gradients:
                lip_pre_clip_norm = check_gradient_magnitudes(self.lip_net_dI)
                self.clip_gradients(self.lip_net_dI, self.opt.disc_clip_value)
                lip_post_clip_norm = check_gradient_magnitudes(self.lip_net_dI)
                self.log(
                    name="lip_img_disc_post", value=lip_post_clip_norm, batch_size=source_clip.shape[0], sync_dist=True
                )
                self.log(
                    name="lip_img_disc_pre", value=lip_pre_clip_norm, batch_size=source_clip.shape[0], sync_dist=True
                )

            opt_lip_d.step()
            self.untoggle_optimizer(opt_lip_d)
        else:
            loss_d_lip = loss_d_face
        loss_d = (loss_d_face + loss_d_lip) * 0.5
        # Store losses for analysis
        # self.disc_losses.append(loss_d_face.item())
        # self.lip_disc_losses.append(loss_d_lip.item())
        # genertor
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()

        _, pred_fake_dI = self.net_dI(fake_out)
        _, pred_fake_lip_dI = self.lip_net_dI(
            fake_out[:, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]]
        )

        loss = {}
        loss["perception_loss"] = self.perception_loss(fake_out, gt_source_clip) * self.opt.lambda_perception
        loss["l1_loss"] = F.l1_loss(fake_out, gt_source_clip) * self.opt.lambda_perception * 5
        loss["perception_lip_loss"] = (
            self.perception_loss(
                fake_out[:, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]],
                gt_source_clip[:, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]],
            )
            * self.opt.lambda_perception
        )
        loss["l1_lip_loss"] = (
            F.l1_loss(
                fake_out[:, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]],
                gt_source_clip[:, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]],
            )
            * self.opt.lambda_perception
            * 5
        )
        loss["sync_loss"] = (
            self.process_sync(fake_out_resized, deep_speech_full.type_as(fake_out)) * self.opt.lambda_syncnet_perception
        )
        if self.opt.identity_loss:
            # we will calculate fake_out with gt_source and reference and take an average.
            loss["identity_loss"] = self.identity_loss.calculate_loss(fake_out, gt_source_clip) * 0.25
        else:
            loss["identity_loss"] = 0
        loss["gen_adv_loss"] = self.gan_loss(pred_fake_dI, True)
        loss["lip_gen_adv_loss"] = self.gan_loss(pred_fake_lip_dI, True)
        loss["total_gen_loss"] = (
            loss["perception_loss"]
            + loss["l1_loss"]
            + loss["perception_lip_loss"]
            + loss["l1_lip_loss"]
            + loss["gen_adv_loss"]
            + (loss["lip_gen_adv_loss"] if False else loss["gen_adv_loss"])
            + loss["identity_loss"]
            + loss["sync_loss"]
        )

        self.manual_backward(loss["total_gen_loss"] + 5 * (loss["l1_lip_loss"] + loss["perception_lip_loss"]))
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

        if batch_idx % 50 == 0 and self.trainer.is_global_zero:
            # gt_source_clip_mask = gt_source_clip.clone().detach()
            # gt_source_clip_mask[
            #     :, :, 112 : 240, 64 : 192
            # ] = 0
            # grid = vis_images_on_wandb(gt_source_clip, gt_source_clip_mask, fake_out, self.opt.batch_size)
            # wgrid = wandb.Image(
            #     grid,
            #     caption=f"loss_g: {loss['total_gen_loss'].item():.4f}, loss_di: {loss['total_disc_loss'].item():.4f}, epoch: {self.current_epoch}",
            # )
            # wandb.log({"images_train": [wgrid]})
            # model_input = F.interpolate(
            #     model_input, size=(256, 256), mode="bilinear"
            # )
            source_clip = rearrange(source_clip, "(t b) c h w -> b c t h w", t=5)
            # gt_source_clip = F.interpolate(
            #     gt_source_clip, size=(256, 256), mode="bilinear"
            # )
            gt_source_clip = rearrange(gt_source_clip, "(t b) c h w -> b c t h w", t=5)
            fake_out = rearrange(fake_out, "(t b) c h w -> b c t h w", t=5)
            create_clip_image_grid(gt_source_clip, gt_source_clip, fake_out, gt_reference_clip, path="train.gif")
            wandb.log({"train_video": [wandb.Video("train.gif")]})
            if self.current_epoch % 10 == 0:
                wandb.alert(title=f"Train epoch: {self.current_epoch}", text=self.dict2string(loss))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_name = self.opt.data[dataloader_idx]["ds_name"]
        (source_clip_v, reference_clip, deep_speech_clip, deep_speech_full, _) = batch
        gt_source_clip = torch.cat(torch.split(source_clip_v, 1, dim=1), 0).squeeze(1)
        gt_reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1)
        deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1)

        if self.opt.upscale > 1:
            source_clip = F.interpolate(gt_source_clip, scale_factor=1 / self.opt.upscale, mode="bilinear")
            # reference_clip = F.interpolate(gt_reference_clip, scale_factor=1 / self.opt.upscale, mode="bilinear")
        else:
            source_clip = gt_source_clip
        reference_clip = gt_reference_clip
        fake_out, reference_clip = self(
            source_clip,
            reference_clip,
            deep_speech_clip,
            mask_dim=self.mask_dim,
        )
        gt_source_clip = source_clip.clone()
        fake_out_resized = F.interpolate(fake_out, scale_factor=1 / self.opt.upscale, mode="bilinear")
        _, pred_fake_dI = self.net_dI(fake_out)
        _, pred_fake_lip_dI = self.lip_net_dI(
            fake_out[:, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]]
        )

        loss = {}
        # Generator loss
        ## insight face loss  TODO
        ## lip distance l1 loss TODO
        loss["perception_loss"] = self.perception_loss(fake_out, gt_source_clip) * self.opt.lambda_perception
        loss["l1_loss"] = F.l1_loss(fake_out, gt_source_clip) * self.opt.lambda_perception * 5
        loss["perception_lip_loss"] = (
            self.perception_loss(
                fake_out[:, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]],
                gt_source_clip[:, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]],
            )
            * self.opt.lambda_perception
        )
        loss["l1_lip_loss"] = (
            F.l1_loss(
                fake_out[:, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]],
                gt_source_clip[:, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]],
            )
            * self.opt.lambda_perception
            * 5
        )
        loss["sync_loss"] = (
            self.process_sync(fake_out_resized, deep_speech_full.type_as(fake_out)) * self.opt.lambda_syncnet_perception
        )
        loss["gen_adv_loss"] = self.gan_loss(pred_fake_dI, True)
        loss["lip_gen_adv_loss"] = self.gan_loss(pred_fake_lip_dI, True)
        if self.opt.identity_loss:
            # we will calculate fake_out with gt_source and reference and take an average.
            loss["identity_loss"] = self.identity_loss.calculate_loss(fake_out, gt_source_clip) * 0.25
        else:
            loss["identity_loss"] = 0
        loss["total_gen_loss"] = (
            loss["perception_loss"]
            + loss["l1_loss"]
            + loss["perception_lip_loss"]
            + loss["l1_lip_loss"]
            + loss["gen_adv_loss"]
            + loss["lip_gen_adv_loss"]
            + loss["identity_loss"]
            + loss["sync_loss"]
        )

        # Discriminator loss
        loss["disc_loss"], _ = self.discriminator(fake_out, gt_source_clip)
        loss["lip_disc_loss"], _ = self.lip_discriminator(fake_out, gt_source_clip)
        loss["total_disc_loss"] = (loss["disc_loss"] + loss["lip_disc_loss"]) * 0.5

        for k, v in loss.items():
            self.log(
                name=f"val_{dataset_name}_{k}",
                value=v,
                prog_bar=True,
                batch_size=self.opt.batch_size,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        if batch_idx == 0 and self.trainer.is_global_zero:
            source_clip = rearrange(source_clip, "(t b) c h w -> b c t h w", t=5)
            gt_source_clip = rearrange(gt_source_clip, "(t b) c h w -> b c t h w", t=5)
            fake_out = rearrange(fake_out, "(t b) c h w -> b c t h w", t=5)
            create_clip_image_grid(
                gt_source_clip, gt_source_clip, fake_out, gt_reference_clip, path=f"val_{dataset_name}.gif"
            )
            wandb.log({f"val_{dataset_name}": [wandb.Video(f"val_{dataset_name}.gif")]})
            if self.current_epoch % 10 == 0:
                wandb.alert(
                    title=f"Validation epoch: {self.current_epoch} dataset: {dataset_name}",
                    text=self.dict2string(loss),
                )

        self.val_loss.append(loss["total_gen_loss"])
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
        # TODO: if image_size is 512, resize it to 256
        fo = F.interpolate(fake_out, size=(256, 256), mode="bilinear")
        fake_out_mouth = fo[:, :, 112 : 240 - 48, 64:192]
        # fake_out_mouth = torch.cat(
        #     [i.unsqueeze(1) for i in torch.split(fake_out_mouth, self.opt.batch_size, dim=0)],
        #     1,
        # )
        fake_out_mouth = rearrange(fake_out_mouth, "(t b) c h w -> b t c h w", t=5)
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
        # TODO: if image_size is 512, then dimensions changes.
        fake_out_mouth = fake_out[:, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]]
        source_image_data_mouth = source_image_data[
            :, :, self.mask_dim[1] : self.mask_dim[3], self.mask_dim[0] : self.mask_dim[2]
        ]
        _, pred_fake_dI = self.lip_net_dI(fake_out_mouth.detach())
        loss_dI_fake = self.gan_loss(pred_fake_dI, False)

        real_features, pred_real_dI = self.lip_net_dI(source_image_data_mouth)
        loss_dI_real = self.gan_loss(pred_real_dI, True)
        loss_dI = (loss_dI_fake + loss_dI_real) * 0.5
        return loss_dI, real_features

    def configure_optimizers(self):
        # Filter out copy mouth parameters from net_g optimization
        net_g_params = []
        for name, param in self.net_g.named_parameters():
            if not name.startswith("copy_mouth."):
                net_g_params.append(param)
        opt_g = optim.AdamW(net_g_params, lr=opt.lr_g, fused=True)
        opt_di = optim.AdamW(self.net_dI.parameters(), lr=opt.lr_dI, fused=True)
        opt_lip_di = optim.AdamW(self.lip_net_dI.parameters(), lr=opt.lr_lip_dI, fused=True)

        self.net_g_scheduler = get_scheduler(opt_g, opt.non_decay, opt.decay, lr_policy=self.opt.scheduler)
        self.net_dI_scheduler = get_scheduler(opt_di, opt.non_decay, opt.decay, lr_policy=self.opt.scheduler)
        self.lip_net_dI_scheduler = get_scheduler(opt_lip_di, opt.non_decay, opt.decay, lr_policy=self.opt.scheduler)
        return [opt_g, opt_di, opt_lip_di], [self.net_g_scheduler, self.net_dI_scheduler, self.lip_net_dI_scheduler]

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

    from lipsync.dinet.ds import load_concat_data

    training_data_loader, val_data_loader = load_concat_data(opt)

    model = DINetClipLightningModule(opt)

    tqdm_callback = TQDMProgressBar(refresh_rate=4)
    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_ep_loss",
        filename="{epoch}-{step}-{val_ep_loss:.3f}-{train_ep_loss:.3f}-{val_ep_sync_loss:.3f}-{train_ep_sync_loss:.3f}",
        save_last=True,
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
    num_devices = opt.devices if isinstance(opt.devices, int) else len(opt.devices)
    trainer = L.Trainer(
        num_nodes=opt.nodes,
        accelerator=opt.accelarator,
        devices=opt.devices,
        max_epochs=opt.non_decay + opt.decay + 1,
        strategy="ddp_find_unused_parameters_true" if num_devices > 1 else "auto",
        num_sanity_val_steps=2,
        sync_batchnorm=True,
        log_every_n_steps=5,
        check_val_every_n_epoch=opt.check_val_every_n_epoch,
        callbacks=[tqdm_callback, lr_callback, checkpoint_callback],
        logger=wand_log,
    )
    if trainer.is_global_zero & (opt.resume_from_checkpoint is not None):
        trainer.fit(
            model,
            train_dataloaders=training_data_loader,
            val_dataloaders=val_data_loader,
            ckpt_path=opt.resume_from_checkpoint,
        )
    else:
        trainer.fit(
            model,
            train_dataloaders=training_data_loader,
            val_dataloaders=val_data_loader,
        )
    # trainer.fit(
    #     model,
    #     train_dataloaders=training_data_loader,
    #     val_dataloaders=val_data_loader,
    # )
