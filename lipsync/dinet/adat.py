import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coordinate_grid_3d(spatial_size, type):
    """
    generate 3D coordinate grid
    """
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)
    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1
    z = 2 * (z / (d - 1)) - 1
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
    return meshed, zz


class AdaAT(nn.Module):
    """
    AdaAT operator
    """

    def __init__(self, para_ch, feature_ch):
        super(AdaAT, self).__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(nn.Linear(para_ch, para_ch), nn.ReLU())
        self.scale = nn.Sequential(nn.Linear(para_ch, feature_ch), nn.Sigmoid())
        self.rotation = nn.Sequential(nn.Linear(para_ch, feature_ch), nn.Tanh())
        self.translation = nn.Sequential(nn.Linear(para_ch, 2 * feature_ch), nn.Tanh())
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map, para_code):
        batch, d, h, w = (
            feature_map.size(0),
            feature_map.size(1),
            feature_map.size(2),
            feature_map.size(3),
        )
        para_code = self.commn_linear(para_code)
        scale = self.scale(para_code).unsqueeze(-1) * 2
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159  #
        rotation_matrix = torch.cat(
            [torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)],
            -1,
        )
        rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)
        grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
        grid_xy = grid_xy.to(feature_map.device)
        grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        grid_z = grid_z.to(feature_map.device)
        grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
        translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
        full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
        trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid, align_corners=False).squeeze(1)
        return trans_feature
