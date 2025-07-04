import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lipsync.dinet.adat import AdaAT
from lipsync.dinet.attention import MultiHeadSpatialAttention
from lipsync.dinet.spade import SPADEDecoder


class AudioProjModel(torch.nn.Module):
    """Audio Projection Model

    This class defines an audio projection model that takes audio embeddings as input
    and produces context tokens as output. The model is based on the ModelMixin class
    and consists of multiple linear layers and activation functions. It can be used
    for various audio processing tasks.

    Attributes:
        seq_len (int): The length of the audio sequence.
        blocks (int): The number of blocks in the audio projection model.
        channels (int): The number of channels in the audio projection model.
        intermediate_dim (int): The intermediate dimension of the model.
        context_tokens (int): The number of context tokens in the output.
        output_dim (int): The output dimension of the context tokens.

    Methods:
        __init__(self, seq_len=5, blocks=12, channels=768, intermediate_dim=512, context_tokens=32, output_dim=768):
            Initializes the AudioProjModel with the given parameters.
        forward(self, audio_embeds):
            Defines the forward pass for the AudioProjModel.
            Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).

    """

    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = torch.nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = torch.nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = torch.nn.Linear(intermediate_dim, output_dim)

        self.norm = torch.nn.LayerNorm(output_dim)

    def forward(self, audio_embeds):
        """
        Defines the forward pass for the AudioProjModel.

        Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).

        Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).
        """
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(batch_size, self.output_dim)

        context_tokens = self.norm(context_tokens)
        #         context_tokens = rearrange(
        #             context_tokens, "(bz f) m c -> bz f m c", f=video_length
        #         )

        return context_tokens


class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.norm = torch.nn.BatchNorm2d(out_features)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SameBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_features)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm2d(out_features)
        self.norm2 = nn.BatchNorm2d(out_features)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = nn.Conv2d(in_features, out_features, 1) if in_features != out_features else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        residual = self.downsample(x)
        return self.activation(out + residual)


class UpBlock2d(nn.Module):
    """
    basic block
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = torch.nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class MappingNetwork(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, hidden_dim=512, num_layers=8):
        super(MappingNetwork, self).__init__()

        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)


class ReferenceFrameEncoder(nn.Module):
    def __init__(self, num_frames=5):
        super(ReferenceFrameEncoder, self).__init__()
        self.num_frames = num_frames

        # Single frame encoder
        self.frame_encoder = nn.Sequential(
            SameBlock2d(3, 32, kernel_size=7, padding=3),
            DownBlock2d(32, 64, kernel_size=3, padding=1),
        )

        # Feature fusion network
        self.fusion_network = nn.Sequential(
            SameBlock2d(64 * num_frames, 128, kernel_size=3, padding=1), DownBlock2d(128, 256, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to separate frames: (B, 5, 3, H, W)
        x = x.view(B, self.num_frames, 3, H, W)

        # Process each frame independently
        frame_features = []
        for i in range(self.num_frames):
            features = self.frame_encoder(x[:, i])
            frame_features.append(features)

        # Concatenate features from all frames
        combined_features = torch.cat(frame_features, dim=1)
        # Final fusion and processing
        output = self.fusion_network(combined_features)
        return output


class SegFormerFaceParser:
    def __init__(self, device="cpu"):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.model.eval()
        self.model.to(device)

        # freeze the params of the model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        size = images.shape[2:]
        image_tensors = self.image_processor(images=images, return_tensors="pt", do_rescale=False).pixel_values.to(
            images.device
        )
        with torch.no_grad():
            outputs = self.model(image_tensors)
            logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)
        upsampled_logits = F.interpolate(
            logits,
            size=(size[0], size[1]),  # H x W
            mode="bilinear",
            align_corners=False,
        )
        return upsampled_logits

    def get_lips_region(self, images):
        upsampled_logits = self.forward(images)
        labels = upsampled_logits.argmax(dim=1)
        mask = torch.zeros_like(labels, dtype=torch.float32)
        mask[labels == 10] = 1
        mask[labels == 11] = 0.5
        mask[labels == 12] = 0.5
        return mask

    def __call__(self, images):
        size = images.shape[2:]
        image_tensors = self.image_processor(images=images, return_tensors="pt", do_rescale=False).pixel_values.to(
            images.device
        )
        with torch.no_grad():
            outputs = self.model(image_tensors)
            logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

        upsampled_logits = F.interpolate(
            logits,
            size=(size[0], size[1]),  # H x W
            mode="bilinear",
            align_corners=False,
        )

        # get label masks
        labels = upsampled_logits.argmax(dim=1)
        labels[labels > 0] = 1
        return labels


def generate_grid_bbox_holes(h, w, length, mask_percent=0.5):
    x = np.arange(0, w, length)
    y = np.arange(0, h, length)
    bbox = []
    for i in x:
        for j in y:
            bbox.append([i, j, i + length, j + length])
    bbox = np.array(bbox)
    # suffle the bbox
    np.random.shuffle(bbox)
    bbox = bbox[: int(len(bbox) * mask_percent)]
    return bbox


class DINetSPADE(nn.Module):
    def __init__(
        self,
        source_channel,
        ref_channel,
        audio_seq_len=5,
        upscale=1,
        reference_frames_process="channel",
        seg_face=False,
        use_attention=False,
    ):
        super(DINetSPADE, self).__init__()
        self.upscale = upscale
        self.seg_face = seg_face
        self.use_attention = use_attention
        self.output_mask = None
        if self.seg_face:
            self.segface = SegFormerFaceParser()
        else:
            self.segface = nn.Identity()

        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )

        self.source_bg_in_conv = nn.Sequential(
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
            SameBlock2d(512, 512, kernel_size=3, padding=1),
            SameBlock2d(512, 512, kernel_size=11, padding=5),
            SameBlock2d(512, 512, kernel_size=11, padding=5),
            DownBlock2d(512, 512, kernel_size=3, padding=1),
            SameBlock2d(512, 512, kernel_size=7, padding=3),
            SameBlock2d(512, 512, kernel_size=7, padding=3),
            DownBlock2d(512, 512, kernel_size=3, padding=1),
            SameBlock2d(512, 512, kernel_size=3, padding=1),
            DownBlock2d(512, 512, kernel_size=3, padding=1),
            SameBlock2d(512, 512, kernel_size=3, padding=1),
            DownBlock2d(512, 512, kernel_size=3, padding=1),
        )
        self.audio_encoder = AudioProjModel(seq_len=audio_seq_len, blocks=12, output_dim=512)

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
        self.adaAT = AdaAT(1024, 256)

        # SPADE decoder
        self.spade_decoder = SPADEDecoder(
            upscale=upscale,
            max_features=256,  # Changed to match adapted channel dimension
            block_expansion=64,
            out_channels=64,
            num_down_blocks=2,
        )

        self.global_avg2d = nn.AdaptiveAvgPool2d(1)

        # Audio feature projection
        self.audio_projection = MappingNetwork(512, 512)

        self.overall_mapping = nn.Sequential(
            nn.Conv2d(1280 if self.seg_face else 1024, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        if self.use_attention:
            self.multihead_attention = MultiHeadSpatialAttention(channels=256, num_heads=4, reduction_ratio=16)
        else:
            self.multihead_attention = nn.Identity()

        # self.copy_mouth = CopyMouth()
        # self.copy_mouth.download_if_no_models()
        # self.copy_mouth.load_models()

    def forward(self, source_img, ref_img, audio_feature, mask_dim):
        source_img = source_img.clone()
        orig_source_img = source_img.clone()
        # source_img = F.interpolate(source_img, (256, 256), mode="bilinear", align_corners=False)
        if self.seg_face:
            logits = self.segface.forward(source_img)
            bg_mask = logits.argmax(dim=1)
            bg_mask[bg_mask > 0] = 1
            bg_mask = bg_mask.unsqueeze(1).repeat(1, 3, 1, 1).detach()
            source_image_bg = source_img * (bg_mask == 0)
            source_image_bg = source_image_bg.detach()
            source_bg_in_feature = self.source_bg_in_conv(source_image_bg)
        source_img[:, :, mask_dim[1] : mask_dim[3], mask_dim[0] : mask_dim[2]] = 0
        source_in_feature = self.source_in_conv(source_img)  # [1, 256, 64, 64]

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
        if self.seg_face:
            merge_feature = torch.cat(
                [source_in_feature, source_bg_in_feature, ref_trans_feature, audio_spatial], 1
            )  # [1, 1024, 64, 64]
        else:
            merge_feature = torch.cat([source_in_feature, ref_trans_feature, audio_spatial], 1)  # [1, 1024, 64, 64]

        # Adapt channel dimension
        merge_feature = self.overall_mapping(merge_feature)
        if self.use_attention:
            merge_feature = self.multihead_attention(merge_feature)
        # Use SPADE decoder
        out = self.spade_decoder(merge_feature)  # Should output [1, 3, 256, 256]
        if self.output_mask is None:
            self.output_mask = torch.zeros_like(out, dtype=torch.float32)
            self.output_mask[:, :, mask_dim[1] : mask_dim[3], mask_dim[0] : mask_dim[2]] = 1
            self.output_mask = F.avg_pool2d(self.output_mask, 7, stride=1, padding=3)
            self.output_mask = self.output_mask[0, 0, :, :].detach()
        out = out * self.output_mask + orig_source_img * (1 - self.output_mask)
        return out, ref_img


if __name__ == "__main__":
    source_img = torch.rand((4, 3, 512, 512))
    ref_img = torch.rand((4, 15, 512, 512))
    audio = torch.randn((4, 5, 12, 768))
    # mask_dim = [64, 112, 192, 240]
    mask_dim = [128, 224, 384, 480]

    model = DINetSPADE(3, 15, audio_seq_len=5, seg_face=True, upscale=1)
    model.eval()
    ff = model(source_img, ref_img, audio, mask_dim)
    for x in ff:
        print(x.shape)

    params = 0
    for _, param in model.named_parameters():
        params += param.numel()
    print(params)  # ~206 million params for audio_seq_length=5 and ref_img of 15
