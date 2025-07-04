from pathlib import Path
from typing import Union

import torch
import torchvision
from torch import nn

from lipsync.identity_loss.iresnet import iresnet100


def load_model(model_path: Union[str, Path], device: str) -> torch.nn.Module:
    """Load a PyTorch model from a file"""
    model = iresnet100(pretrained=False, fp16=False)
    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    return model


class IdentityLoss(nn.Module):
    def __init__(self, checkpoint_path: Union[str, Path], device: str):
        super().__init__()
        self.model = load_model(checkpoint_path, device)
        self.loss_fn = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        x = torchvision.transforms.Resize((112, 112))(x)
        if x.max() > 1:
            x = x / 255.0
        x = 2 * x - 1
        return self.model(x)

    def calculate_loss(self, x1, x2):
        # cat x1 and x2 and calculate loss
        # visualize x1 images in a grid
        x1_vec = self.forward(x1)
        x2_vec = self.forward(x2)
        sim = self.loss_fn(x1_vec, x2_vec)
        sim = torch.clamp(sim, min=torch.finfo(sim.dtype).eps, max=1 - torch.finfo(sim.dtype).eps)
        loss_sim = -1 * torch.mean(torch.log(sim))
        return loss_sim


if __name__ == "__main__":
    import fastcore.all as fc
    import numpy as np
    import torchvision

    model = IdentityLoss("weights/arcface.pt", "cuda")
    print(model)

    folders = fc.L(fc.Path("/data/prakash_lipsync/video_hallo/hdtf").glob("*"))
    x1 = fc.L(folders[np.random.randint(0, len(folders))].glob("*"))
    x2 = fc.L(folders[np.random.randint(0, len(folders))].glob("*"))

    imgs1 = fc.L(x1[np.random.randint(0, len(x1))].glob("*.png"))
    imgs1.sort()
    imgs2 = fc.L(x2[np.random.randint(0, len(x2))].glob("*.png"))
    imgs2.sort()

    imgs1 = [torchvision.io.read_image(img) for img in imgs1]
    imgs2 = [torchvision.io.read_image(img) for img in imgs2]
    imgs = torch.stack(imgs1 + imgs2)
    imgs = imgs.to("cuda")
    vec = model(imgs)
    # between user img1 and user img2
    sim1 = torch.zeros(len(imgs1), len(imgs2))
    for i in range(len(imgs1)):
        for j in range(len(imgs2)):
            sim1[i, j] = torch.nn.functional.cosine_similarity(vec[i], vec[j + len(imgs1)], dim=0)
    print(sim1)

    # between user img1
    sim2 = torch.zeros(len(imgs1), len(imgs1))
    for i in range(len(imgs1)):
        for j in range(len(imgs1)):
            sim2[i, j] = torch.nn.functional.cosine_similarity(vec[i], vec[j], dim=0)
    print(sim2)

    # between user img2
    sim3 = torch.zeros(len(imgs2), len(imgs2))
    for i in range(len(imgs2)):
        for j in range(len(imgs2)):
            sim3[i, j] = torch.nn.functional.cosine_similarity(vec[i + len(imgs1)], vec[j + len(imgs1)], dim=0)
    print(sim3)

    # get Non-diagnoal values max
    sim2_non_diag = sim2[~torch.eye(sim2.shape[0], sim2.shape[1], dtype=bool)].max()
    sim3_non_diag = sim3[~torch.eye(sim3.shape[0], sim3.shape[1], dtype=bool)].max()

    print(sim1.min(), sim2_non_diag.max(), sim3_non_diag.max())
    # the best way to check is pick any random images and check against each of them.
