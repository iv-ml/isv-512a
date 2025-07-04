# Source image and predicted image keypoints are calculated and L1 loss is calculated.
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from lipsync.appearence_loss.convnextv2 import convnextv2_tiny


class AppearanceLoss(nn.Module):
    def __init__(self, weights_path):
        super().__init__()

        self.model = convnextv2_tiny()
        self.load_weights(weights_path)
        self.model.eval()
        self.model.to("cuda")

    def load_weights(self, weights_path):
        weights = load_file(weights_path)
        detector_weights = {k.replace("detector.", ""): v for k, v in weights.items() if "detector." in k}
        state_dict = self.model.state_dict()
        state_dict.update(detector_weights)
        self.model.load_state_dict(state_dict)

    def forward(self, source_image, predicted_image):
        source_kp = self.model(source_image)
        predicted_kp = self.model(predicted_image)
        return F.l1_loss(source_kp, predicted_kp)


if __name__ == "__main__":
    weights = load_file("/home/prakash/lipsync/weights/appearance_feature_extractor.safetensors")

    model = convnextv2_tiny()
    model.load_state_dict(weights)
    model.eval()
    model.to("cuda")
    breakpoint()
