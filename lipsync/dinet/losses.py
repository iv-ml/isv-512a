# we will divide the losses into 3 parts:
# Discriminator loss 
#    - full image loss
#    - mouth region loss

# appearance loss
#    - reconstruction loss (L1 loss)
#    - perceptual loss (VGG16 features)


import lpips
import torch
import torch.nn as nn


class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        self.lpips = lpips.LPIPS(net="vgg")

    def forward(self, generated, target):
        return self.lpips(generated, target).mean()


class GANLoss(nn.Module):
    """
    GAN loss
    """

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)