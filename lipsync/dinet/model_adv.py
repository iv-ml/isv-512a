import torch
import torch.nn as nn
import torch.nn.functional as F

from lipsync.dinet.adat import AdaAT
from lipsync.dinet.model import (
    AudioProjModel,
    DownBlock2d,
    MappingNetwork,
    ReferenceFrameEncoder,
    ResBlock2d,
    SameBlock2d,
    SegFormerFaceParser,
)
from lipsync.dinet.spade import SPADEDecoder2


class DINetSPADE(nn.Module):
    def __init__(
        self,
        source_channel,
        ref_channel,
        audio_seq_len=5,
        upscale=1,
        lip_upscale=1,
        reference_frames_process="channel",
        seg_face=False,
    ):
        super(DINetSPADE, self).__init__()
        if seg_face:
            source_channel = source_channel * 2
            self.segface = SegFormerFaceParser()
        else:
            self.segface = nn.Identity()

        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        if reference_frames_process == "channel":
            self.ref_in_conv = nn.Sequential(
                SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
                DownBlock2d(64, 128, kernel_size=3, padding=1),
                DownBlock2d(128, 256, kernel_size=3, padding=1),
            )
        elif reference_frames_process == "video":
            self.ref_in_conv = ReferenceFrameEncoder(num_frames=ref_channel // 3)
        else:
            raise ValueError(f"Invalid reference frames process: {reference_frames_process}")

        self.trans_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.audio_encoder = AudioProjModel(seq_len=audio_seq_len, blocks=12, output_dim=256)

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(384, 256)

        # Add channel adaptation layer before SPADE decoder
        # self.channel_adapt = nn.Conv2d(1024, 256, kernel_size=1)

        # SPADE decoder
        self.face_decoder = SPADEDecoder2(
            upscale=upscale,
            max_features=256,  # Changed to match adapted channel dimension
            block_expansion=64,
            out_channels=64,
            num_down_blocks=2,
        )

        self.lip_decoder = SPADEDecoder2(
            upscale=lip_upscale,
            max_features=256,  # Changed to match adapted channel dimension
            block_expansion=64,
            out_channels=64,
            num_down_blocks=2,
        )

        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

        # Audio feature projection
        self.audio_projection = MappingNetwork(256, 256)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

        # Add a three layer cnn network to map features from 768 to 256
        self.overall_mapping = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, source_img, ref_img, audio_feature, mask_dim):
        source_img = source_img.clone()
        bg_mask = self.segface(source_img)
        if len(bg_mask.shape) == 3:
            bg_mask = bg_mask.unsqueeze(1)
            bg_mask = bg_mask.repeat(1, 3, 1, 1)
            source_image_bg = source_img * (bg_mask == 0)
            source_img[:, :, mask_dim[1] : mask_dim[3], mask_dim[0] : mask_dim[2]] = 0
            source_img_2 = torch.concat([source_img, source_image_bg], axis=1)
        else:
            source_img[:, :, mask_dim[1] : mask_dim[3], mask_dim[0] : mask_dim[2]] = 0
            source_img_2 = source_img
        source_in_feature = self.source_in_conv(source_img_2)  # [1, 256, 64, 64]
        # if visualize:
        #     import torchvision
        #     breakpoint()
        #     x = torchvision.transforms.ToPILImage()((255*(2*source_img)-1).to(torch.uint8)[0])
        #     x.save("temp/source_in_feature.png")
        ref_in_feature = self.ref_in_conv(ref_img)  # [1, 256, 64, 64]

        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)

        audio_proj = self.audio_encoder(audio_feature)
        trans_para = torch.cat([img_para, audio_proj], 1)
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)

        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)

        audio_spatial = self.audio_projection(audio_proj)
        audio_spatial = audio_spatial + audio_proj  # Residual connection
        audio_spatial = F.leaky_relu(audio_spatial, 0.2)
        audio_spatial = (
            audio_spatial.unsqueeze(2).unsqueeze(2).expand(-1, -1, source_in_feature.size(2), source_in_feature.size(3))
        )

        merge_feature = torch.cat([source_in_feature, ref_trans_feature, audio_spatial], 1)  # [1, 768, 64, 64]
        merge_feature = self.overall_mapping(merge_feature)

        # Adapt channel dimension
        # adapted_feature = self.channel_adapt(merge_feature)  # [1, 256, 64, 64]

        # Use SPADE decoder
        lip_out, mask_features = self.lip_decoder(merge_feature, replace_feature=None, replace_dim=None)
        # do bilinear interpolation to match the size of the mask
        # mask_dim_resized = [i // 2 for i in mask_dim]
        h, w = mask_dim[3] - mask_dim[1], mask_dim[2] - mask_dim[0]
        mask_features = F.interpolate(mask_features, size=(h, w), mode="bilinear", align_corners=False)
        face_out, _ = self.face_decoder(
            merge_feature, replace_feature=mask_features, replace_dim=mask_dim
        )  # Should output [1, 3, 256, 256]
        return face_out, lip_out


class DINetSPADE2(nn.Module):
    def __init__(
        self,
        source_channel,
        ref_channel,
        audio_seq_len=5,
        upscale=1,
        lip_upscale=1,
        reference_frames_process="channel",
        seg_face=True,
    ):
        super(DINetSPADE2, self).__init__()
        self.segface = SegFormerFaceParser()

        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )

        self.source_seg_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        if reference_frames_process == "channel":
            self.ref_in_conv = nn.Sequential(
                SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
                DownBlock2d(64, 128, kernel_size=3, padding=1),
                DownBlock2d(128, 256, kernel_size=3, padding=1),
            )
        # elif reference_frames_process == "video":
        #     self.ref_in_conv = ReferenceFrameEncoder(num_frames=ref_channel // 3)
        else:
            raise ValueError(f"Invalid reference frames process: {reference_frames_process}")

        self.trans_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.audio_encoder = AudioProjModel(seq_len=audio_seq_len, blocks=12, output_dim=256)

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(384, 256)

        # Add channel adaptation layer before SPADE decoder
        # self.channel_adapt = nn.Conv2d(1024, 256, kernel_size=1)

        # SPADE decoder
        self.face_decoder = SPADEDecoder2(
            upscale=upscale,
            max_features=256,  # Changed to match adapted channel dimension
            block_expansion=64,
            out_channels=64,
            num_down_blocks=2,
        )

        self.lip_decoder = SPADEDecoder2(
            upscale=lip_upscale,
            max_features=256,  # Changed to match adapted channel dimension
            block_expansion=64,
            out_channels=64,
            num_down_blocks=2,
        )

        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

        # Audio feature projection
        self.audio_projection = MappingNetwork(256, 256)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

        # Add a three layer cnn network to map features from 768 to 256
        self.overall_mapping = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, source_img, ref_img, audio_feature, mask_dim):
        source_img = source_img.clone()
        bg_mask = self.segface(source_img)
        # if len(bg_mask.shape) == 3:
        bg_mask = bg_mask.unsqueeze(1)
        bg_mask = bg_mask.repeat(1, 3, 1, 1)
        source_image_bg = source_img * (bg_mask == 0)
        source_img[:, :, mask_dim[1] : mask_dim[3], mask_dim[0] : mask_dim[2]] = 0
        source_in_feature = self.source_in_conv(source_img)  # [1, 256, 64, 64]
        source_seg_in_feature = self.source_seg_in_conv(source_image_bg)
        # if visualize:
        #     import torchvision
        #     breakpoint()
        #     x = torchvision.transforms.ToPILImage()((255*(2*source_img)-1).to(torch.uint8)[0])
        #     x.save("temp/source_in_feature.png")
        ref_in_feature = self.ref_in_conv(ref_img)  # [1, 256, 64, 64]

        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)

        audio_proj = self.audio_encoder(audio_feature)
        trans_para = torch.cat([img_para, audio_proj], 1)
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)

        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)

        audio_spatial = self.audio_projection(audio_proj)
        audio_spatial = audio_spatial + audio_proj  # Residual connection
        audio_spatial = F.leaky_relu(audio_spatial, 0.2)
        audio_spatial = (
            audio_spatial.unsqueeze(2).unsqueeze(2).expand(-1, -1, source_in_feature.size(2), source_in_feature.size(3))
        )

        merge_feature = torch.cat(
            [source_in_feature, ref_trans_feature, audio_spatial, source_seg_in_feature], 1
        )  # [1, 768, 64, 64]
        merge_feature = self.overall_mapping(merge_feature)
        # Adapt channel dimension
        # adapted_feature = self.channel_adapt(merge_feature)  # [1, 256, 64, 64]

        # Use SPADE decoder
        mask_dim_resized = [i // 4 for i in mask_dim]
        lip_out, mask_features = self.lip_decoder(
            merge_feature[:, :, mask_dim_resized[0] : mask_dim_resized[2], mask_dim_resized[1] : mask_dim_resized[3]],
            replace_feature=None,
            replace_dim=None,
        )
        # do bilinear interpolation to match the size of the mask
        # mask_dim_resized = [i // 2 for i in mask_dim]
        # h, w = mask_dim[3] - mask_dim[1], mask_dim[2] - mask_dim[0]
        # mask_features = F.interpolate(mask_features, size=(h, w), mode="bilinear", align_corners=False)
        face_out, _ = self.face_decoder(
            merge_feature, replace_feature=mask_features, replace_dim=mask_dim
        )  # Should output [1, 3, 256, 256]
        return face_out, lip_out


if __name__ == "__main__":
    source_img = torch.rand((1, 3, 256, 256))
    ref_img = torch.rand((1, 15, 256, 256))
    audio = torch.randn((1, 5, 12, 768))

    model = DINetSPADE2(3, 15, audio_seq_len=5, upscale=2, lip_upscale=1, seg_face=True)
    model.eval()
    face_out, lip_out = model(source_img, ref_img, audio, mask_dim=[64, 112, 192, 240])
    print(face_out.shape, lip_out.shape)

    params = 0
    for _, param in model.named_parameters():
        params += param.numel()
    print(params)  # 159,109,071
