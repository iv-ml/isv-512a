import torch
import torch.nn.functional as F
from lipsync.syncnet.model import SyncNetMulti

class SyncNetPerceptionMulti(torch.nn.Module):
    def __init__(self, weight_loc):
        super().__init__()
        k = torch.load(weight_loc)
        self.cfg, weights = k["params"], k["state_dict"]
        self.net = SyncNetMulti(
            out_dim=self.cfg["out_dim"], backbone_name=self.cfg["backbone_name"]
        )
        self.net.load_state_dict(weights)
        self.net.cuda()
        self.net.eval()
        self.ss = torch.nn.CosineSimilarity()

    def forward(self, img, aud):
        with torch.no_grad():
            img_fe, audio_fe = self.net(img, aud)
        a = F.relu(audio_fe)
        v = F.relu(img_fe)
        a = F.normalize(a, p=2, dim=1)
        v = F.normalize(v, p=2, dim=1)
        cs = self.ss(v, a)
        return cs