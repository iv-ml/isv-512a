from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch import nn


def conv3x3(in_planes: int, out_planes: int, strd: int = 1, padding: int = 1, bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias)


class HDTR_ConvBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        self.bn1 = nn.SyncBatchNorm(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.SyncBatchNorm(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = nn.Sequential(nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual
        return out3


class HDTR_D_ConvBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        name: str,
        nums: int = 3,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.nums = nums
        self.relu = nn.ReLU(True)
        if not isinstance(name, str):
            raise ValueError("name should be str")
        self.name = name

        for i in range(self.nums):
            setattr(
                self,
                f"conv{name}_{i}",
                nn.Conv2d(inplanes, outplanes, padding=padding, kernel_size=kernel_size, stride=stride),
            )
            setattr(self, f"conv{name}_{i}_bn", nn.BatchNorm2d(outplanes))
            inplanes = outplanes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = x
        for i in range(self.nums):
            net = getattr(self, f"conv{self.name}_{i}")(net)
            net = getattr(self, f"conv{self.name}_{i}_bn")(net)
            net = self.relu(net)
        return net


class HDTR_HourGlass(nn.Module):
    def __init__(self, num_modules: int, depth: int, num_features: int):
        super().__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.dropout = nn.Dropout(0.5)
        self._generate_network(self.depth)

    def _generate_network(self, level: int):
        self.add_module(f"b1_{level}", HDTR_ConvBlock(256, 256))
        self.add_module(f"b2_{level}", HDTR_ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module(f"b2_plus_{level}", HDTR_ConvBlock(256, 256))

        self.add_module(f"b3_{level}", HDTR_ConvBlock(256, 256))

    def _forward(self, level: int, inp: torch.Tensor) -> torch.Tensor:
        up1 = inp
        up1 = self._modules[f"b1_{level}"](up1)
        up1 = self.dropout(up1)

        low1 = F.max_pool2d(inp, 2, stride=1)
        low1 = self._modules[f"b2_{level}"](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules[f"b2_plus_{level}"](low2)

        low3 = low2
        low3 = self._modules[f"b3_{level}"](low3)
        up1size = up1.size()
        up2 = F.interpolate(low3, size=(up1size[2], up1size[3]), mode="bilinear", align_corners=True)

        return up1 + up2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(self.depth, x)


class HDTR_FAN_use(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.num_modules = 1

        # Base part
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = HDTR_ConvBlock(64, 128)
        self.conv3 = HDTR_ConvBlock(128, 128)
        self.conv4 = HDTR_ConvBlock(128, 256)

        # Stacking part
        self.m0 = HDTR_HourGlass(1, 3, 256)
        self.top_m_0 = HDTR_ConvBlock(256, 256)
        self.conv_last0 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.add_module("l0", nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        self.add_module("bn_end0", nn.BatchNorm2d(256))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.max_pool2d(self.conv2(x), 1)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        hg = self.m0(previous)
        ll = hg
        ll = self.top_m_0(ll)
        ll = self.bn_end0(self.conv_last0(ll))
        tmp_out = self.l0(F.relu(ll))

        return tmp_out


class HDTR_FanFusion(nn.Module):
    def __init__(self, mask_channels: int = 3, ref_channels: int = 3):
        super().__init__()
        self.mask_model = HDTR_FAN_use(mask_channels)
        self.ref_model = HDTR_FAN_use(ref_channels)

        self.m_conv1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1),
        )
        self.m_conv2 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1),
        )

    def forward(
        self, mask: torch.Tensor, mean_mask: torch.Tensor, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_in = torch.cat((mask, mean_mask), 1)
        net1 = self.mask_model(mask_in)
        net2 = self.ref_model(image)

        net1_out = self.m_conv1(net1)
        net2_out = self.m_conv2(net2)

        return net1_out, net2_out


class HDTR_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.deconv1_1_new = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.deconv1_1_bn = nn.BatchNorm2d(512)
        self.convblock1 = HDTR_D_ConvBlock(512, 256, "1", nums=2)
        self.convblock2 = HDTR_D_ConvBlock(256, 128, "2", nums=3)
        self.convblock3 = HDTR_D_ConvBlock(128, 64, "3", nums=4)
        self.conv4_1 = nn.ConvTranspose2d(64, 32, 3, 1, 1)
        self.conv4_1_bn = nn.BatchNorm2d(32)
        self.conv4_2 = nn.ConvTranspose2d(32, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.deconv1_1_new(x)
        net = self.relu(self.deconv1_1_bn(net))
        for i in range(3):
            net = getattr(self, f"convblock{i + 1}")(net)
            net = self.upsample(net)
        net = self.conv4_1(net)
        net = self.relu(self.conv4_1_bn(net))
        net = self.conv4_2(net)
        net = self.tanh(net)
        net = (net + 1) / 2.0
        return net


class HDTR_Generator(nn.Module):
    def __init__(self, mask_channels: int = 3, ref_channels: int = 3):
        super(HDTR_Generator, self).__init__()
        self.encoder = HDTR_FanFusion(mask_channels, ref_channels)
        self.decoder = HDTR_Decoder()

    def forward(self, mask, mean_mask, image):
        out_net1, out_net2 = self.encoder.forward(mask, mean_mask, image)

        encoder_f = torch.cat((out_net1, out_net2), 1)
        out_g = self.decoder.forward(encoder_f)
        return out_g


class HDTR_GANLoss(nn.Module):
    def __init__(self, use_lsgan: bool = True, target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def get_target_tensor(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class HDTR_PerceptionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg19(pretrained=True)
        self.vgg.eval()
        self.features = self.vgg.features
        self.feature_layers = ["4", "9", "18", "27", "36"]
        self.mse_loss = nn.MSELoss()

    def getfeatures(self, x: torch.Tensor) -> list:
        feature_list = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.feature_layers:
                feature_list.append(x)
        return feature_list

    def __call__(self, video_pd: torch.Tensor, video_gt: torch.Tensor) -> torch.Tensor:
        features_pd = self.getfeatures(video_pd)
        features_gt = self.getfeatures(video_gt)

        with torch.no_grad():
            features_gt = [x.detach() for x in features_gt]

        perceptual_loss = sum([self.mse_loss(features_pd[i], features_gt[i]) for i in range(len(features_gt))])
        return perceptual_loss


class HDTR_nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class HDTR_Mouth_disc_qual(nn.Module):
    def __init__(self, size):
        super(HDTR_Mouth_disc_qual, self).__init__()
        self.size = size
        assert self.size == 128, "HDTR_Mouth_disc_qual only supports size 128"
        self.face_encoder_blocks_ = nn.ModuleList(
            [
                nn.Sequential(HDTR_nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)),  # 128
                nn.Sequential(
                    HDTR_nonorm_Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 64
                    HDTR_nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                ),
                nn.Sequential(
                    HDTR_nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 32
                    HDTR_nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
                ),
                nn.Sequential(
                    HDTR_nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 16
                    HDTR_nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
                ),
                nn.Sequential(
                    HDTR_nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 8
                    HDTR_nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                ),
                nn.Sequential(
                    HDTR_nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 4
                    HDTR_nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                ),
                nn.Sequential(
                    HDTR_nonorm_Conv2d(512, 512, kernel_size=4, stride=1, padding=0),  # 1
                    HDTR_nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),
            ]
        )

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, mouth):  # mouth: [b, 3, 96, 96]
        x = mouth
        for f in self.face_encoder_blocks_:
            x = f(x)
        return self.binary_pred(x).view(len(x), -1)


class SegFormerFaceParser(nn.Module):
    def __init__(self, device):
        super().__init__()
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
        n_images = image_tensors.size(0)
        bs = 100
        outputs = []
        for i in range(0, n_images, bs):
            with torch.no_grad():
                outputs.append(self.model(image_tensors[i : i + bs]).logits)
        logits = torch.cat(outputs, dim=0)
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


def get_batch_binary_midpoints(tensor):
    batch_size, H, W = tensor.size()
    batch_indices = torch.nonzero(tensor)

    # If no 1s at all, return image midpoint for all batches
    if batch_indices.size(0) == 0:
        image_midpoint = (H / 2, W / 2)
        return [image_midpoint] * batch_size

    # Forward pass: use previous valid midpoint as default
    midpoints = []
    prev_midpoint = None

    for b in range(batch_size):
        batch_mask = batch_indices[:, 0] == b
        indices = batch_indices[batch_mask][:, 1:]

        if indices.size(0) == 0:  # No 1s in this batch
            midpoints.append(prev_midpoint)  # Could be None
            continue

        min_coords = torch.min(indices, dim=0)[0]
        max_coords = torch.max(indices, dim=0)[0]
        midpoint = (min_coords + max_coords) / 2
        curr_midpoint = (midpoint[0].item(), midpoint[1].item())

        midpoints.append(curr_midpoint)
        prev_midpoint = curr_midpoint

    # Backward pass: fill in None values with next valid midpoint
    for b in range(batch_size - 2, -1, -1):  # Go backwards excluding last element
        if midpoints[b] is None:
            midpoints[b] = midpoints[b + 1]

    return midpoints


class HDTR(nn.Module):
    def __init__(self, size=96, inference=False):
        super().__init__()
        self.size = size
        self.face_parser = SegFormerFaceParser(device="cuda")
        self.generator = HDTR_Generator(6, 3)
        self.inference = inference

    def forward(self, source_image, ref_image):
        batch_size = source_image.size(0)
        si = source_image.clone()
        si = F.interpolate(si, (512, 512), mode="bilinear", align_corners=False)
        si = F.pad(si, (self.size // 2, self.size // 2, self.size // 2, self.size // 2), "constant", 0)
        teeth_mask = self.face_parser.get_lips_region(si)
        midpoints = get_batch_binary_midpoints(teeth_mask > 0)
        # self.size = 10
        teeth_mask = teeth_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        teeth_images = []
        new_masks = []
        for i, mp in enumerate(midpoints):
            a = round(mp[1]) - self.size // 2
            b = round(mp[0]) - self.size // 2
            new_mask = teeth_mask[i, :, b : b + self.size, a : a + self.size]
            new_masks.append(new_mask)
            teeth_image = si[i, :, b : b + self.size, a : a + self.size]
            teeth_images.append(teeth_image)
        teeth_images = torch.stack(teeth_images, dim=0)
        new_masks = torch.stack(new_masks, dim=0)

        ri = ref_image.clone()
        ri = F.interpolate(ri, (512, 512), mode="bilinear", align_corners=False)
        ri = rearrange(ri, "b (f c) h w -> (b f) c h w", c=3)
        ri = F.pad(ri, (self.size // 2, self.size // 2, self.size // 2, self.size // 2), "constant", 0)
        ref_mask = self.face_parser.get_lips_region(ri)
        mouth_openness = (ref_mask == 1).sum(dim=(1, 2))
        mouth_openness = rearrange(mouth_openness, "(b f) -> b f", b=batch_size)
        ref_mids = get_batch_binary_midpoints(ref_mask > 0)
        ref_images = []
        for i, mp in enumerate(ref_mids):
            a = round(mp[1]) - self.size // 2
            b = round(mp[0]) - self.size // 2
            ref_image = ri[i, :, b : b + self.size, a : a + self.size]
            ref_images.append(ref_image)
        ref_images = torch.stack(ref_images, dim=0)
        ref_images = rearrange(ref_images, "(b f) c h w -> b f c h w", b=batch_size)
        batch_indices = torch.arange(ref_images.size(0), device=ref_images.device)
        selected_indices = mouth_openness.argmax(dim=1)
        selected_refs = ref_images[batch_indices, selected_indices, ...]
        model_input = (teeth_images * (new_masks == 0)).detach()
        output = self.generator(model_input, new_masks.detach(), selected_refs.detach())
        output = output * (model_input == 0) + model_input * (model_input != 0)
        if self.inference:
            for i, mp in enumerate(midpoints):
                a = round(mp[1]) - self.size // 2
                b = round(mp[0]) - self.size // 2
                si[i, :, b : b + self.size, a : a + self.size] = output[i, :, :, :]
            # remove padding from si
            return si[:, :, self.size // 2 : -self.size // 2, self.size // 2 : -self.size // 2]
        return model_input, new_masks, selected_refs, output, teeth_images


if __name__ == "__main__":
    from pathlib import Path

    import torchvision.utils as vutils
    from PIL import Image

    img_folder = Path("/data/lipsync_512_data/iv_recording/C1038--0001/000032")
    images = list(img_folder.glob("*.png"))
    images.sort()
    images = [Image.open(img) for img in images]
    images = [np.array(img) for img in images]
    images = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in images]
    images = torch.stack(images, dim=0).cuda()
    model = HDTR(size=128).cuda()
    refs = torch.cat([images, images, images, images, images], dim=0)
    model_input, new_masks, ref_images, output, gt = model(images, refs)
    # teeth_images and new_masks are 5x3x96x96
    # concat them horizontally to create teeth and mask images using make grid
    # then concat them vertically to create the final image and save it
    ti = vutils.make_grid(gt, nrow=5, padding=2)
    mi = vutils.make_grid(new_masks, nrow=5, padding=2)
    ri = vutils.make_grid(ref_images, nrow=5, padding=2)
    oi = vutils.make_grid(output, nrow=5, padding=2)
    mi1 = vutils.make_grid(model_input, nrow=5, padding=2)
    final_image = torch.cat((mi1, mi, ri, oi, ti), 1)
    vutils.save_image(final_image, "final_image.png")
    # discriminator
    discriminator = HDTR_Mouth_disc_qual(size=128).cuda()
    print(f"output shape: {discriminator(output).shape}")
    print(f"gt shape: {discriminator(gt).shape}")
    # ganloss
    gan_loss = HDTR_GANLoss(use_lsgan=True, target_real_label=1.0, target_fake_label=0.0).cuda()
    loss = gan_loss(discriminator(output), True)
    print(f"gan loss: {loss}")
    # perception loss
    perception_loss = HDTR_PerceptionLoss().cuda()
    loss = perception_loss(output, gt)
    print(f"perception loss: {loss}")
