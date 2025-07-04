import copy
import os
from typing import Optional

import numpy as np
import safetensors.torch
import torch
from loguru import logger

from live_portrait.modules.config.inference_config import InferenceConfig
from live_portrait.modules.live_portrait.appearance_feature_extractor import AppearanceFeatureExtractor
from live_portrait.modules.live_portrait.live_portrait_wrapper import LivePortraitWrapper
from live_portrait.modules.live_portrait.model_downloader import MODELS_URL, download_model
from live_portrait.modules.live_portrait.motion_extractor import MotionExtractor
from live_portrait.modules.live_portrait.spade_generator import SPADEDecoder
from live_portrait.modules.live_portrait.stitching_retargeting_network import StitchingRetargetingNetwork
from live_portrait.modules.live_portrait.warping_network import WarpingNetwork
from live_portrait.modules.utils.camera import get_rotation_matrix
from live_portrait.modules.utils.helper import load_yaml
from live_portrait.modules.utils.paths import MODEL_CONFIG, MODELS_DIR


class ExpressionSet:
    def __init__(self, erst=None, es=None, batch_size=1, device=None):
        self.device = device
        self.batch_size = batch_size
        if es is not None:
            self.e = copy.deepcopy(es.e)  # [:, :, :]
            self.r = copy.deepcopy(es.r)  # [:]
            self.s = copy.deepcopy(es.s)
            self.t = copy.deepcopy(es.t)
        elif erst is not None:
            self.e = erst[0]
            self.r = erst[1]
            self.s = erst[2]
            self.t = erst[3]
        else:
            self.e = torch.from_numpy(np.zeros((self.batch_size, 21, 3))).float().to(self.device)
            self.r = torch.Tensor([0, 0, 0])
            self.s = 0
            self.t = 0

    def div(self, value):
        self.e /= value
        self.r /= value
        self.s /= value
        self.t /= value

    def add(self, other):
        self.e += other.e
        self.r += other.r
        self.s += other.s
        self.t += other.t

    def sub(self, other):
        self.e -= other.e
        self.r -= other.r
        self.s -= other.s
        self.t -= other.t

    def mul(self, value):
        self.e *= value
        self.r *= value
        self.s *= value
        self.t *= value


class CopyMouth(torch.nn.Module):
    def __init__(self, device: Optional[str] = None):
        super(CopyMouth, self).__init__()
        self.model_dir = MODELS_DIR
        self.model_config = load_yaml(MODEL_CONFIG)["model_params"]
        self.pipeline = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def load_models(self):
        logger.info("Loading models...")
        logger.info("Loading appearance feature extractor...")
        appearance_feat_config = self.model_config["appearance_feature_extractor_params"]
        self.appearance_feature_extractor = AppearanceFeatureExtractor(**appearance_feat_config).to(self.device)
        self.appearance_feature_extractor = self.load_safe_tensor(
            self.appearance_feature_extractor, os.path.join(self.model_dir, "appearance_feature_extractor.safetensors")
        )

        logger.info("Loading motion extractor...")
        motion_ext_config = self.model_config["motion_extractor_params"]
        self.motion_extractor = MotionExtractor(**motion_ext_config).to(self.device)
        self.motion_extractor = self.load_safe_tensor(
            self.motion_extractor, os.path.join(self.model_dir, "motion_extractor.safetensors")
        )

        logger.info("Loading warping module...")
        warping_module_config = self.model_config["warping_module_params"]
        self.warping_module = WarpingNetwork(**warping_module_config).to(self.device)
        self.warping_module = self.load_safe_tensor(
            self.warping_module, os.path.join(self.model_dir, "warping_module.safetensors")
        )

        logger.info("Loading spade generator...")
        spaded_decoder_config = self.model_config["spade_generator_params"]
        self.spade_generator = SPADEDecoder(**spaded_decoder_config).to(self.device)
        self.spade_generator = self.load_safe_tensor(
            self.spade_generator, os.path.join(self.model_dir, "spade_generator.safetensors")
        )

        logger.info("Loading stitching retargeting module...")
        stitcher_config = self.model_config["stitching_retargeting_module_params"]
        self.stitching_retargeting_module = StitchingRetargetingNetwork(**stitcher_config.get("stitching")).to(
            self.device
        )
        self.stitching_retargeting_module = self.load_safe_tensor(
            self.stitching_retargeting_module,
            os.path.join(self.model_dir, "stitching_retargeting_module.safetensors"),
            True,
        )

        if self.pipeline is None:
            logger.info("Loading live portrait wrapper...")
            self.pipeline = LivePortraitWrapper(
                InferenceConfig(),
                self.appearance_feature_extractor,
                self.motion_extractor,
                self.warping_module,
                self.spade_generator,
                {"stitching": self.stitching_retargeting_module},
            )

    def forward(self, source_image, ref_image, inference_type="mouth"):
        # we need to copy ref image lip position to source image.
        # source image is B, 3, H, W, ref image is B, 3, H, W
        # mapping happens on 1->1 basis
        source_image = source_image.to(self.device)
        ref_image = ref_image.to(self.device)
        source_x_s_info = self.pipeline.get_kp_info(source_image)
        source_f_s_user = self.pipeline.extract_feature_3d(source_image)
        source_x_s_user = self.pipeline.transform_keypoint(source_x_s_info)

        src_ratio = 1.0  # Im not sure what src ratio actually is. #TODO
        s_exp = source_x_s_info["exp"] * src_ratio
        s_exp[:, 5] = source_x_s_info["exp"][:, 5]
        s_exp += source_x_s_info["kp"]

        ref_x_s_info = self.pipeline.get_kp_info(ref_image)
        # ref_f_s_user = self.pipeline.extract_feature_3d(ref_image)
        # ref_x_s_user = self.pipeline.transform_keypoint(ref_x_s_info)

        s_exp = source_x_s_info["exp"] + source_x_s_info["kp"]
        d_exp = ref_x_s_info["exp"] + source_x_s_info["kp"]

        es = ExpressionSet(batch_size=source_image.shape[0], device=self.device)
        factor = 1.0  # not sure what happens if we change this. #TODO
        if inference_type == "mouth":
            idxes = (14, 17, 19, 20)
        elif inference_type == "expression":
            idxes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
        elif inference_type == "expression_but_not_mouth":
            idxes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18)
        else:
            raise ValueError(f"Invalid inference type: {inference_type}")
        for idx in range(es.e.shape[1]):
            if idx in idxes:
                es.e[:, idx] = d_exp[:, idx] * factor
            else:
                es.e[:, idx] = s_exp[:, idx]

        new_rotate = get_rotation_matrix(source_x_s_info["pitch"], source_x_s_info["yaw"], source_x_s_info["roll"])

        x_d_new = (source_x_s_info["scale"] * (1 + es.s)).unsqueeze(2) * ((es.e) @ new_rotate) + source_x_s_info[
            "t"
        ].unsqueeze(1)

        x_d_new = self.pipeline.stitching(source_x_s_user, x_d_new)

        crop_out = self.pipeline.warp_decode(source_f_s_user, source_x_s_user, x_d_new)
        crop_out = torch.nn.functional.interpolate(
            crop_out["out"], size=(source_image.shape[2], source_image.shape[3]), mode="bilinear", align_corners=False
        )
        return crop_out

    def download_if_no_models(self):
        models_urls_dic = MODELS_URL
        model_dir = self.model_dir

        for model_name, model_url in models_urls_dic.items():
            if model_url.endswith(".pt"):
                model_name += ".pt"
            elif model_url.endswith(".n2x"):
                model_name += ".n2x"
            else:
                model_name += ".safetensors"
            model_path = os.path.join(model_dir, model_name)
            if not os.path.exists(model_path):
                download_model(model_path, model_url)

    @staticmethod
    def load_safe_tensor(model, file_path, is_stitcher=False):
        def filter_stitcher(checkpoint, prefix):
            filtered_checkpoint = {
                key.replace(prefix + "_module.", ""): value
                for key, value in checkpoint.items()
                if key.startswith(prefix)
            }
            return filtered_checkpoint

        if is_stitcher:
            model.load_state_dict(filter_stitcher(safetensors.torch.load_file(file_path), "retarget_shoulder"))
        else:
            model.load_state_dict(safetensors.torch.load_file(file_path))
        model.eval()
        return model


if __name__ == "__main__":
    import fastcore.all as fc
    import torchvision

    copy_mouth = CopyMouth()
    copy_mouth.download_if_no_models()
    copy_mouth.load_models()

    root = fc.Path("/data/prakash_lipsync/video_hallo_256_16_mini/vfhq/")
    videos = fc.L([i.name for i in root.glob("*")])

    # video = videos[np.random.randint(0, len(videos))]
    # video = videos[0]
    # images = fc.L((root/video).rglob("*.png"))
    # images.sort()

    # img1 = images[0]
    # img2 = images[1]

    # img1t = torchvision.io.read_image(img1, mode=torchvision.io.ImageReadMode.RGB)/255.
    # img2t = torchvision.io.read_image(img2, mode=torchvision.io.ImageReadMode.RGB)/255.

    # copy_mouth(img1t[None], img2t[None])

    for _ in range(1000):
        t1 = []
        t2 = []
        for i in range(8):
            video = videos[np.random.randint(0, len(videos))]
            images = fc.L((root / video).rglob("*.png"))
            images.sort()
            img1 = images[np.random.randint(0, len(images))]
            img2 = images[np.random.randint(0, len(images))]

            img1t = torchvision.io.read_image(img1, mode=torchvision.io.ImageReadMode.RGB) / 255.0
            img2t = torchvision.io.read_image(img2, mode=torchvision.io.ImageReadMode.RGB) / 255.0
            t1.append(img1t[None])
            t2.append(img2t[None])

        t1 = torch.concat(t1)
        t2 = torch.concat(t2)

        out = copy_mouth(t1, t2)
