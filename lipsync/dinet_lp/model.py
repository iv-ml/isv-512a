import torch
import torch.nn as nn
import torch.nn.functional as F

from lipsync.dinet.adat import AdaAT
from lipsync.dinet.model import (
    AudioProjModel,
    DownBlock2d,
    MappingNetwork,
    ResBlock2d,
    SameBlock2d,
)
from lipsync.dinet.spade import SPADEResnetBlock


class SPADEDecoder(nn.Module):
    def __init__(
        self,
        upscale=1,
        max_features=256,
        block_expansion=64,
        out_channels=64,
        num_down_blocks=2,
    ):
        for i in range(num_down_blocks):
            input_channels = min(max_features, block_expansion * (2 ** (i + 1)))
        self.upscale = upscale
        super().__init__()
        norm_G = "spadespectralinstance"
        label_num_channels = input_channels  # 256

        self.fc = nn.Conv2d(input_channels, 2 * input_channels, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_4 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_5 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.up_0 = SPADEResnetBlock(2 * input_channels, input_channels, norm_G, label_num_channels)
        self.up_1 = SPADEResnetBlock(input_channels, out_channels, norm_G, label_num_channels)
        self.up = nn.Upsample(scale_factor=2)

        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
        else:
            self.conv_img = nn.Sequential(
                nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2),
            )

    def forward(self, feature, replace_feature=None, replace_mask=None):
        seg = feature  # Bx256x64x64
        x = self.fc(feature)  # Bx512x64x64
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)

        x = self.up(x)  # Bx512x64x64 -> Bx512x128x128
        x = self.up_0(x, seg)  # Bx512x128x128 -> Bx256x128x128
        x = self.up(x)  # Bx256x128x128 -> Bx256x256x256
        feature = self.up_1(x, seg)  # Bx256x256x256 -> Bx64x256x256
        if replace_feature is not None:
            # replace mask has 1 outside and 0 inside
            feature = torch.where(replace_mask == 0, replace_feature, feature)
            # feature[:, :, replace_dim[1]:replace_dim[3], replace_dim[0]:replace_dim[2]] = replace_feature
        x = self.conv_img(F.leaky_relu(feature, 2e-1))  # Bx64x256x256 -> Bx3xHxW
        x = torch.sigmoid(x)  # Bx3xHxW

        return x, feature


class DINetSPADE(nn.Module):
    def __init__(
        self,
        source_channel,
        ref_channel,
        audio_seq_len=5,
        upscale=1,
    ):
        super(DINetSPADE, self).__init__()

        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )

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
        self.face_decoder = SPADEDecoder(
            upscale=upscale,
            max_features=256,  # Changed to match adapted channel dimension
            block_expansion=64,
            out_channels=64,
            num_down_blocks=2,
        )

        # self.lip_decoder = SPADEDecoder(
        #     upscale=lip_upscale,
        #     max_features=256,  # Changed to match adapted channel dimension
        #     block_expansion=64,
        #     out_channels=64,
        #     num_down_blocks=2,
        # )

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

    def forward(self, source_img, ref_img, audio_feature, source_mask):
        source_img = source_img.clone()
        source_img = torch.where(source_mask > 0.5, source_img, torch.zeros_like(source_img))
        source_in_feature = self.source_in_conv(source_img)  # [1, 256, 64, 64]

        # torchvision.transforms.ToPILImage()(source_img[0])
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

        face_out, _ = self.face_decoder(
            merge_feature, replace_feature=None, replace_mask=None
        )  # Should output [1, 3, 256, 256]
        return face_out


if __name__ == "__main__":
    source_img = torch.rand((4, 3, 256, 256))
    ref_img = torch.rand((4, 15, 256, 256))
    audio = torch.randn((4, 5, 12, 768))
    source_mask = torch.ones((4, 1, 256, 256))
    source_mask[:, :, 112:192, 96:240] = 0

    model = DINetSPADE(3, 15, audio_seq_len=5, upscale=2)
    model.eval()
    face_out = model(source_img, ref_img, audio, source_mask)
    print(face_out.shape)

    params = 0
    for _, param in model.named_parameters():
        params += param.numel()
    print(params)  # 159,109,071
