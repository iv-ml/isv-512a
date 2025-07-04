import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.leaky_relu(out, 0.2)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=4, stride=2, padding=1):
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_features, out_features, kernel_size, stride, padding)
        )

    def forward(self, x):
        return F.leaky_relu(self.conv(x), 0.2)


class EnhancedDiscriminator(nn.Module):
    def __init__(
        self, num_channels, block_expansion=64, num_blocks=4, max_features=512
    ):
        super(EnhancedDiscriminator, self).__init__()

        self.initial = DiscriminatorBlock(
            num_channels, block_expansion, stride=1, padding=1
        )

        down_blocks = []
        feature_size = block_expansion
        for i in range(num_blocks):
            down_blocks.append(
                DiscriminatorBlock(
                    feature_size, min(max_features, block_expansion * (2 ** (i + 1)))
                )
            )
            feature_size = min(max_features, block_expansion * (2 ** (i + 1)))

        self.down_blocks = nn.ModuleList(down_blocks)

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(feature_size) for _ in range(2)]
        )

        self.final_conv = nn.utils.spectral_norm(
            nn.Conv2d(feature_size, 1, kernel_size=1)
        )

    def forward(self, x):
        feature_maps = []
        out = self.initial(x)
        # feature_maps.append(out)

        for down_block in self.down_blocks:
            out = down_block(out)
            feature_maps.append(out)

        for res_block in self.residual_blocks:
            out = res_block(out)

        out = self.final_conv(out)
        # feature_maps.append(out)

        return feature_maps, out
    
if __name__ == "__main__":
    model = EnhancedDiscriminator(3, block_expansion=64, num_blocks=4, max_features=512)
    params = 0
    for _, param in model.named_parameters():
        params += param.numel()
    print(params) #16 million 
