# we will use resnet18 for now - it will have take 15 channels as input
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ImageEncoder(nn.Module):
    def __init__(self, in_channels=15):
        super(ImageEncoder, self).__init__()

        # Load a pre-trained ResNet18 model
        pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept 15 channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize the new conv1 layer with the average of the pre-trained weights
        with torch.no_grad():
            new_weight = pretrained_model.conv1.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            self.conv1.weight.copy_(new_weight)

        # Copy the rest of the pre-trained layers
        self.bn1 = pretrained_model.bn1
        self.relu = pretrained_model.relu
        self.maxpool = pretrained_model.maxpool
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4
        self.avgpool = pretrained_model.avgpool

        # Remove the final fully connected layer
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ImageEncoder()
    print(model)
    x = torch.randn(4, 15, 256, 256)
    print(model(x).shape)
