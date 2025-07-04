import logging
import os
import cv2
import time
import copy
import dill
import torch
from ultralytics import YOLO
import safetensors.torch
import gradio as gr
from loguru import logger
from gradio_i18n import Translate, gettext as _
from ultralytics.utils import LOGGER as ultralytics_logger
from enum import Enum
from typing import Union

from live_portrait.modules.utils.paths import *
from live_portrait.modules.utils.image_helper import *
from live_portrait.modules.live_portrait.model_downloader import *
from live_portrait.modules.live_portrait.live_portrait_wrapper import LivePortraitWrapper
from live_portrait.modules.utils.camera import get_rotation_matrix
from live_portrait.modules.utils.helper import load_yaml
from live_portrait.modules.utils.constants import *
from live_portrait.modules.config.inference_config import InferenceConfig
from live_portrait.modules.live_portrait.spade_generator import SPADEDecoder
from live_portrait.modules.live_portrait.warping_network import WarpingNetwork
from live_portrait.modules.live_portrait.motion_extractor import MotionExtractor
from live_portrait.modules.live_portrait.appearance_feature_extractor import AppearanceFeatureExtractor
from live_portrait.modules.live_portrait.stitching_retargeting_network import StitchingRetargetingNetwork


class LivePortraitInferencer:
    def __init__(self, model_dir: str = MODELS_DIR, output_dir: str = OUTPUTS_DIR, gpu_id: int = 0):
        self.model_dir = model_dir
        os.makedirs(os.path.join(self.model_dir, "animal"), exist_ok=True)
        self.output_dir = output_dir
        self.model_config = load_yaml(MODEL_CONFIG)["model_params"]

        self.appearance_feature_extractor = None
        self.motion_extractor = None
        self.warping_module = None
        self.spade_generator = None
        self.stitching_retargeting_module = None
        self.pipeline = None
        self.detect_model = None
        self.gpu_id = gpu_id
        self.device = self.get_device(gpu_id)
        self.model_type = ModelType.HUMAN.value

        self.mask_img = None
        self.temp_img_idx = 0
        self.src_image = None
        self.src_image_list = None
        self.sample_image = None
        self.driving_images = None
        self.driving_values = None
        self.crop_factor = None
        self.psi = None
        self.psi_list = None
        self.d_info = None

    def load_models(self, model_type: str = ModelType.HUMAN.value, progress=gr.Progress()):
        if isinstance(model_type, ModelType):
            model_type = model_type.value
        if model_type not in [mode.value for mode in ModelType]:
            model_type = ModelType.HUMAN.value

        self.model_type = model_type
        if model_type == ModelType.ANIMAL.value:
            model_dir = os.path.join(self.model_dir, "animal")
        else:
            model_dir = self.model_dir

        self.download_if_no_models(model_type=model_type)

        total_models_num = 5
        progress(0 / total_models_num, desc="Loading Appearance Feature Extractor model...")
        appearance_feat_config = self.model_config["appearance_feature_extractor_params"]
        self.appearance_feature_extractor = AppearanceFeatureExtractor(**appearance_feat_config).to(self.device)
        self.appearance_feature_extractor = self.load_safe_tensor(
            self.appearance_feature_extractor, os.path.join(model_dir, "appearance_feature_extractor.safetensors")
        )

        progress(1 / total_models_num, desc="Loading Motion Extractor model...")
        motion_ext_config = self.model_config["motion_extractor_params"]
        self.motion_extractor = MotionExtractor(**motion_ext_config).to(self.device)
        self.motion_extractor = self.load_safe_tensor(
            self.motion_extractor, os.path.join(model_dir, "motion_extractor.safetensors")
        )

        progress(2 / total_models_num, desc="Loading Warping Module model...")
        warping_module_config = self.model_config["warping_module_params"]
        self.warping_module = WarpingNetwork(**warping_module_config).to(self.device)
        self.warping_module = self.load_safe_tensor(
            self.warping_module, os.path.join(model_dir, "warping_module.safetensors")
        )

        progress(3 / total_models_num, desc="Loading Spade generator model...")
        spaded_decoder_config = self.model_config["spade_generator_params"]
        self.spade_generator = SPADEDecoder(**spaded_decoder_config).to(self.device)
        self.spade_generator = self.load_safe_tensor(
            self.spade_generator, os.path.join(model_dir, "spade_generator.safetensors")
        )

        progress(4 / total_models_num, desc="Loading Stitcher model...")
        stitcher_config = self.model_config["stitching_retargeting_module_params"]
        self.stitching_retargeting_module = StitchingRetargetingNetwork(**stitcher_config.get("stitching")).to(
            self.device
        )
        self.stitching_retargeting_module = self.load_safe_tensor(
            self.stitching_retargeting_module, os.path.join(model_dir, "stitching_retargeting_module.safetensors"), True
        )
        self.stitching_retargeting_module = {"stitching": self.stitching_retargeting_module}

        if self.pipeline is None or model_type != self.model_type:
            self.pipeline = LivePortraitWrapper(
                InferenceConfig(),
                self.appearance_feature_extractor,
                self.motion_extractor,
                self.warping_module,
                self.spade_generator,
                self.stitching_retargeting_module,
            )

        det_model_name = "yolo_v5s_animal_det" if model_type == ModelType.ANIMAL else "face_yolov8n"
        self.detect_model = YOLO(MODEL_PATHS[det_model_name]).to(self.device)

    def neutral_expression(self, src_image, model_type: str = ModelType.HUMAN.value, crop_factor=2.5):
        if isinstance(model_type, ModelType):
            model_type = model_type.value
        if model_type not in [mode.value for mode in ModelType]:
            model_type = ModelType.HUMAN

        if self.pipeline is None or model_type != self.model_type:
            self.load_models(model_type=model_type)
        self.crop_factor = crop_factor
        self.psi = self.prepare_source(src_image, crop_factor)
        self.src_image = src_image
        psi = self.psi
        s_info = psi.x_s_info
        s_exp = s_info["kp"]

        new_rotate = get_rotation_matrix(s_info["pitch"], s_info["yaw"], s_info["roll"])
        x_d_new = s_info["scale"] * ((s_exp) @ new_rotate) + s_info["t"]

        x_d_new = self.pipeline.stitching(psi.x_s_user, x_d_new)

        crop_out = self.pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d_new)
        crop_out = self.pipeline.parse_output(crop_out["out"])[0]
        crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR)
        out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(np.uint8)
        return out

    def copy_mouth(
        self,
        model_type: str = ModelType.HUMAN.value,
        rotate_pitch=0,
        rotate_yaw=0,
        rotate_roll=0,
        blink=0,
        eyebrow=0,
        wink=0,
        pupil_x=0,
        pupil_y=0,
        aaa=0,
        eee=0,
        woo=0,
        smile=0,
        src_ratio=1,
        sample_ratio=1,
        sample_parts="All",
        crop_factor=1.5,
        src_image=None,
        sample_image=None,
        motion_link=None,
        add_exp=None,
        pre_cropped=False,
    ):
        if isinstance(model_type, ModelType):
            model_type = model_type.value
        if model_type not in [mode.value for mode in ModelType]:
            model_type = ModelType.HUMAN

        if self.pipeline is None or model_type != self.model_type:
            self.load_models(model_type=model_type)

        try:
            rotate_yaw = -rotate_yaw

            new_editor_link = None
            if isinstance(motion_link, np.ndarray) and motion_link:
                self.psi = motion_link[0]
                new_editor_link = motion_link.copy()
            elif src_image is not None:
                if id(src_image) != id(self.src_image) or self.crop_factor != crop_factor:
                    self.crop_factor = crop_factor
                    self.psi = self.prepare_source(src_image, crop_factor)
                    self.src_image = src_image
                new_editor_link = []
                new_editor_link.append(self.psi)
            else:
                return None

            psi = self.psi
            s_info = psi.x_s_info
            # delta_new = copy.deepcopy()
            s_exp = s_info["exp"] * src_ratio
            s_exp[0, 5] = s_info["exp"][0, 5]
            s_exp += s_info["kp"]

            es = ExpressionSet(gpu_id=self.gpu_id)
            d_psi = self.prepare_source(sample_image, crop_factor, pre_cropped=pre_cropped)
            expression_less = False
            if expression_less:
                d_exp = s_info["kp"]
                s_exp = s_info["kp"]
            else:
                s_exp = s_info["exp"] + s_info["kp"]
                d_exp = d_psi.x_s_info["exp"] + s_info["kp"]
            if isinstance(sample_image, np.ndarray) and sample_image is not None and sample_image.size > 0:
                if self.sample_image is None or id(self.sample_image) != id(sample_image):
                    self.sample_image = sample_image
                    # d_image_np = (sample_image * 255).byte().numpy() if isinstance(sample_image, torch.Tensor) else sample_image
                    d_face = self.crop_face(self.sample_image, 1.7)
                    i_d = self.prepare_src_image(d_face)
                    self.d_info = self.pipeline.get_kp_info(i_d)
                    self.d_info["exp"][0, 5, 0] = 0
                    self.d_info["exp"][0, 5, 1] = 0

            # "OnlyExpression", "OnlyRotation", "OnlyMouth", "OnlyEyes", "All"
            self.copying(
                es.e, s_exp, d_exp, sample_ratio, (14, 17, 19, 20)
            )  # copy mouth from sample image to source image

            new_rotate = get_rotation_matrix(
                s_info["pitch"] + es.r[0], s_info["yaw"] + es.r[1], s_info["roll"] + es.r[2]
            ).to(self.device)  # Add .to(self.device)

            # Ensure tensors are on the same device
            s_exp = s_exp.to(self.device)
            es.e = es.e.to(self.device)

            x_d_new = (s_info["scale"] * (1 + es.s)) * ((s_exp + es.e) @ new_rotate) + s_info["t"]

            x_d_new = self.pipeline.stitching(psi.x_s_user, x_d_new)

            crop_out = self.pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d_new)
            crop_out = self.pipeline.parse_output(crop_out["out"])[0]

            crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR)
            out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(np.uint8)

            temp_out_img_path, out_img_path = (
                get_auto_incremental_file_path(TEMP_DIR, "png"),
                get_auto_incremental_file_path(OUTPUTS_DIR, "png"),
            )
            save_image(numpy_array=crop_out, output_path=temp_out_img_path)
            save_image(numpy_array=out, output_path=out_img_path)

            new_editor_link.append(es)

            # get expression set info for src_image and sample_image
            d_face = self.crop_face(self.src_image, 1.7)
            i_d = self.prepare_src_image(d_face)
            self.d_info = self.pipeline.get_kp_info(i_d)
            self.d_info["exp"][0, 5, 0] = 0
            self.d_info["exp"][0, 5, 1] = 0
            source_exp_info = self.d_info["exp"]
            reference_exp_info = es.e

            return out, source_exp_info, reference_exp_info
        except Exception as e:
            logger.error(f"Error in copy_mouth: {e}")
            raise

    def edit_expression(
        self,
        model_type: str = ModelType.HUMAN.value,
        rotate_pitch=0,
        rotate_yaw=0,
        rotate_roll=0,
        blink=0,
        eyebrow=0,
        wink=0,
        pupil_x=0,
        pupil_y=0,
        aaa=0,
        eee=0,
        woo=0,
        smile=0,
        src_ratio=1,
        sample_ratio=1,
        sample_parts="All",
        crop_factor=1.5,
        src_image=None,
        sample_image=None,
        motion_link=None,
        add_exp=None,
    ):
        try:
            rotate_yaw = -rotate_yaw

            # Ensure models are loaded and on correct device
            if self.pipeline is None or model_type != self.model_type:
                self.load_models(model_type=model_type)

            # Initialize editor link
            new_editor_link = None
            if isinstance(motion_link, np.ndarray) and motion_link:
                self.psi = motion_link[0]
                new_editor_link = motion_link.copy()
            elif src_image is not None:
                if id(src_image) != id(self.src_image) or self.crop_factor != crop_factor:
                    self.crop_factor = crop_factor
                    self.psi = self.prepare_source(src_image, crop_factor)
                    self.src_image = src_image
                new_editor_link = []
                new_editor_link.append(self.psi)
            else:
                return None

            psi = self.psi
            s_info = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in psi.x_s_info.items()}
            
            # Move tensors to correct device
            s_exp = (s_info["exp"] * src_ratio).to(self.device)
            s_exp[0, 5] = s_info["exp"][0, 5].to(self.device)
            s_exp += s_info["kp"].to(self.device)

            es = ExpressionSet(gpu_id=self.gpu_id)

            if isinstance(sample_image, np.ndarray) and sample_image is not None and sample_image.size > 0:
                if self.sample_image is None or id(self.sample_image) != id(sample_image):
                    self.sample_image = sample_image
                    d_face = self.crop_face(self.sample_image, 1.7)
                    i_d = self.prepare_src_image(d_face)
                    self.d_info = self.pipeline.get_kp_info(i_d)
                    self.d_info["exp"][0, 5, 0] = 0
                    self.d_info["exp"][0, 5, 1] = 0

                if sample_parts in [SamplePart.ONLY_EXPRESSION.value, SamplePart.ALL.value]:
                    es.e += self.d_info["exp"].to(self.device) * sample_ratio
                if sample_parts == SamplePart.ONLY_ROTATION.value or sample_parts == SamplePart.ONLY_ROTATION.ALL.value:
                    rotate_pitch += self.d_info["pitch"] * sample_ratio
                    rotate_yaw += self.d_info["yaw"] * sample_ratio
                    rotate_roll += self.d_info["roll"] * sample_ratio
                elif sample_parts == SamplePart.ONLY_MOUTH.value:
                    self.retargeting(es.e, self.d_info["exp"], sample_ratio, (14, 17, 19, 20))
                elif sample_parts == SamplePart.ONLY_EYES.value:
                    self.retargeting(es.e, self.d_info["exp"], sample_ratio, (1, 2, 11, 13, 15, 16))
                elif sample_parts == SamplePart.MOUTH_AND_EYES.value:
                    self.retargeting(es.e, self.d_info["exp"], sample_ratio, (14, 17, 19, 20))
                    self.retargeting(es.e, self.d_info["exp"], sample_ratio, (1, 2, 11, 13, 15, 16))
                elif sample_parts == SamplePart.FULL_FACE.value:
                    es.e = self.d_info["exp"]

            es.r = self.calc_fe(
                es.e,
                blink,
                eyebrow,
                wink,
                pupil_x,
                pupil_y,
                aaa,
                eee,
                woo,
                smile,
                rotate_pitch,
                rotate_yaw,
                rotate_roll,
            ).to(self.device)

            if isinstance(add_exp, ExpressionSet):
                es.add(add_exp)

            # Ensure rotation matrix is on correct device
            new_rotate = get_rotation_matrix(
                s_info["pitch"] + es.r[0],
                s_info["yaw"] + es.r[1],
                s_info["roll"] + es.r[2]
            ).to(self.device)

            # Calculate new expression
            x_d_new = (s_info["scale"] * (1 + es.s)) * ((s_exp + es.e) @ new_rotate) + s_info["t"].to(self.device)
            x_d_new = self.pipeline.stitching(psi.x_s_user.to(self.device), x_d_new)

            # Generate output
            crop_out = self.pipeline.warp_decode(
                psi.f_s_user.to(self.device),
                psi.x_s_user.to(self.device),
                x_d_new
            )
            crop_out = self.pipeline.parse_output(crop_out["out"])[0]

            # Final image processing
            crop_with_fullsize = cv2.warpAffine(
                crop_out,
                psi.crop_trans_m,
                get_rgb_size(psi.src_rgb),
                cv2.INTER_LINEAR
            )
            out = np.clip(
                psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb,
                0,
                255
            ).astype(np.uint8)

            # Save outputs
            temp_out_img_path = get_auto_incremental_file_path(TEMP_DIR, "png")
            out_img_path = get_auto_incremental_file_path(OUTPUTS_DIR, "png")
            save_image(numpy_array=crop_out, output_path=temp_out_img_path)
            save_image(numpy_array=out, output_path=out_img_path)

            new_editor_link.append(es)

            # Get expression info
            d_face = self.crop_face(self.src_image, 1.7)
            i_d = self.prepare_src_image(d_face)
            self.d_info = self.pipeline.get_kp_info(i_d)
            self.d_info["exp"][0, 5, 0] = 0
            self.d_info["exp"][0, 5, 1] = 0
            source_exp_info = self.d_info["exp"].to(self.device)
            reference_exp_info = es.e

            return out, source_exp_info, reference_exp_info

        except Exception as e:
            logger.error(f"Error in edit_expression: {e}")
            raise

    def create_video(
        self,
        retargeting_eyes,
        retargeting_mouth,
        turn_on,
        tracking_src_vid,
        animate_without_vid,
        command,
        crop_factor,
        src_images=None,
        driving_images=None,
        motion_link=None,
        progress=gr.Progress(),
    ):
        if not turn_on:
            return None, None
        src_length = 1

        if src_images is None:
            if motion_link is not None:
                self.psi_list = [motion_link[0]]
            else:
                return None, None

        if src_images is not None:
            src_length = len(src_images)
            if id(src_images) != id(self.src_images) or self.crop_factor != crop_factor:
                self.crop_factor = crop_factor
                self.src_images = src_images
                if 1 < src_length:
                    self.psi_list = self.prepare_source(src_images, crop_factor, True, tracking_src_vid)
                else:
                    self.psi_list = [self.prepare_source(src_images, crop_factor)]

        cmd_list, cmd_length = self.parsing_command(command, motion_link, self.gpu_id)
        if cmd_list is None:
            return None, None
        cmd_idx = 0

        driving_length = 0
        if driving_images is not None:
            if id(driving_images) != id(self.driving_images):
                self.driving_images = driving_images
                self.driving_values = self.prepare_driving_video(driving_images)
            driving_length = len(self.driving_values)

        total_length = max(driving_length, src_length)

        if animate_without_vid:
            total_length = max(total_length, cmd_length)

        c_i_es = ExpressionSet()
        c_o_es = ExpressionSet()
        d_0_es = None
        out_list = []

        psi = None
        for i in range(total_length):
            if i < src_length:
                psi = self.psi_list[i]
                s_info = psi.x_s_info
                s_es = ExpressionSet(
                    erst=(s_info["kp"] + s_info["exp"], torch.Tensor([0, 0, 0]), s_info["scale"], s_info["t"])
                )

            new_es = ExpressionSet(es=s_es)

            if i < cmd_length:
                cmd = cmd_list[cmd_idx]
                if 0 < cmd.change:
                    cmd.change -= 1
                    c_i_es.add(cmd.es)
                    c_i_es.sub(c_o_es)
                elif 0 < cmd.keep:
                    cmd.keep -= 1

                new_es.add(c_i_es)

                if cmd.change == 0 and cmd.keep == 0:
                    cmd_idx += 1
                    if cmd_idx < len(cmd_list):
                        c_o_es = ExpressionSet(es=c_i_es)
                        cmd = cmd_list[cmd_idx]
                        c_o_es.div(cmd.change)
            elif 0 < cmd_length:
                new_es.add(c_i_es)

            if i < driving_length:
                d_i_info = self.driving_values[i]
                d_i_r = torch.Tensor(
                    [d_i_info["pitch"], d_i_info["yaw"], d_i_info["roll"]]
                )  # .float().to(device="cuda:0")

                if d_0_es is None:
                    d_0_es = ExpressionSet(erst=(d_i_info["exp"], d_i_r, d_i_info["scale"], d_i_info["t"]))

                    self.retargeting(s_es.e, d_0_es.e, retargeting_eyes, (11, 13, 15, 16))
                    self.retargeting(s_es.e, d_0_es.e, retargeting_mouth, (14, 17, 19, 20))

                new_es.e += d_i_info["exp"] - d_0_es.e
                new_es.r += d_i_r - d_0_es.r
                new_es.t += d_i_info["t"] - d_0_es.t

            r_new = get_rotation_matrix(
                s_info["pitch"] + new_es.r[0], s_info["yaw"] + new_es.r[1], s_info["roll"] + new_es.r[2]
            )
            d_new = new_es.s * (new_es.e @ r_new) + new_es.t
            d_new = self.pipeline.stitching(psi.x_s_user, d_new)
            crop_out = self.pipeline.warp_decode(psi.f_s_user, psi.x_s_user, d_new)
            crop_out = self.pipeline.parse_output(crop_out["out"])[0]

            crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR)
            out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(np.uint8)
            out_list.append(out)

            progress(i / total_length, "predicting..")

        if len(out_list) == 0:
            return None

        out_imgs = torch.cat([pil2tensor(img_rgb) for img_rgb in out_list])
        return out_imgs

    def download_if_no_models(
        self,
        model_type: str = ModelType.HUMAN.value,
        progress=gr.Progress(),
    ):
        progress(0, desc="Downloading models...")

        if isinstance(model_type, ModelType):
            model_type = model_type.value
        if model_type == ModelType.ANIMAL.value:
            models_urls_dic = MODELS_ANIMAL_URL
            model_dir = os.path.join(self.model_dir, "animal")
        else:
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

    @staticmethod
    def get_device(gpu_id: int = 0):
        if torch.cuda.is_available():
            return f"cuda:{gpu_id}"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def get_temp_img_name(self):
        self.temp_img_idx += 1
        return "expression_edit_preview" + str(self.temp_img_idx) + ".png"

    @staticmethod
    def parsing_command(command, motoin_link, gpu_id: int = 0):
        command.replace(" ", "")
        lines = command.split("\n")

        cmd_list = []

        total_length = 0

        i = 0
        for line in lines:
            i += 1
            if not line:
                continue
            try:
                cmds = line.split("=")
                idx = int(cmds[0])
                if idx == 0:
                    es = ExpressionSet(gpu_id=gpu_id)  # Pass gpu_id here too
                else:
                    es = ExpressionSet(es=motion_link[idx], gpu_id=gpu_id)  # And here
                cmds = cmds[1].split(":")
                change = int(cmds[0])
                keep = int(cmds[1])
            except Exception as e:
                print(f"(AdvancedLivePortrait) Command Err Line {i}: {line}, :{e}")
                return None, None

            total_length += change + keep
            es.div(change)
            cmd_list.append(Command(es, change, keep))

        return cmd_list, total_length

    def get_face_bboxes(self, image_rgb):
        pred = self.detect_model(image_rgb, conf=0.7, device=self.device)
        return pred[0].boxes.xyxy.cpu().numpy()

    def detect_face(self, image_rgb, crop_factor, sort=True):
        original_logger_level = ultralytics_logger.level
        ultralytics_logger.setLevel(logging.CRITICAL + 1)

        bboxes = self.get_face_bboxes(image_rgb)
        w, h = get_rgb_size(image_rgb)

        # print(f"w, h:{w, h}")

        cx = w / 2
        min_diff = w
        best_box = None
        for x1, y1, x2, y2 in bboxes:
            bbox_w = x2 - x1
            if bbox_w < 30:
                continue
            diff = abs(cx - (x1 + bbox_w / 2))
            if diff < min_diff:
                best_box = [x1, y1, x2, y2]
                # print(f"diff, min_diff, best_box:{diff, min_diff, best_box}")
                min_diff = diff

        if best_box == None:
            logger.error("Failed to detect face!!")
            raise Exception("Failed to detect face!!")

        x1, y1, x2, y2 = best_box

        # for x1, y1, x2, y2 in bboxes:
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        crop_w = bbox_w * crop_factor
        crop_h = bbox_h * crop_factor

        crop_w = max(crop_h, crop_w)
        crop_h = crop_w

        kernel_x = int(x1 + bbox_w / 2)
        kernel_y = int(y1 + bbox_h / 2)

        new_x1 = int(kernel_x - crop_w / 2)
        new_x2 = int(kernel_x + crop_w / 2)
        new_y1 = int(kernel_y - crop_h / 2)
        new_y2 = int(kernel_y + crop_h / 2)

        if not sort:
            return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0
        elif w < new_x2:
            new_x1 -= new_x2 - w
            new_x2 = w
            if new_x1 < 0:
                new_x2 -= new_x1
                new_x1 = 0

        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0
        elif h < new_y2:
            new_y1 -= new_y2 - h
            new_y2 = h
            if new_y1 < 0:
                new_y2 -= new_y1
                new_y1 = 0

        if w < new_x2 and h < new_y2:
            over_x = new_x2 - w
            over_y = new_y2 - h
            over_min = min(over_x, over_y)
            new_x2 -= over_min
            new_y2 -= over_min

        ultralytics_logger.setLevel(original_logger_level)
        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

    @staticmethod
    def retargeting(delta_out, driving_exp, factor, idxes):
        for idx in idxes:
            # delta_out[0, idx] -= src_exp[0, idx] * factor
            delta_out[0, idx] += driving_exp[0, idx] * factor

    @staticmethod
    def copying(delta_out, src_exp, driving_exp, factor, idxes):
        for idx in range(delta_out.shape[1]):
            if idx in idxes:
                delta_out[0, idx] = driving_exp[0, idx] * factor
            else:
                delta_out[0, idx] = src_exp[0, idx]

    @staticmethod
    def calc_face_region(square, dsize):
        region = copy.deepcopy(square)
        is_changed = False
        if dsize[0] < region[2]:
            region[2] = dsize[0]
            is_changed = True
        if dsize[1] < region[3]:
            region[3] = dsize[1]
            is_changed = True

        return region, is_changed

    @staticmethod
    def expand_img(rgb_img, square):
        crop_trans_m = create_transform_matrix(max(-square[0], 0), max(-square[1], 0), 1, 1)
        new_img = cv2.warpAffine(
            rgb_img, crop_trans_m, (square[2] - square[0], square[3] - square[1]), cv2.INTER_LINEAR
        )
        return new_img

    def prepare_src_image(self, img):
        h, w = img.shape[:2]
        input_shape = [256, 256]
        if h != input_shape[0] or w != input_shape[1]:
            if 256 < h:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_LINEAR
            x = cv2.resize(img, (input_shape[0], input_shape[1]), interpolation=interpolation)
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.0  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.0  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f"img ndim should be 3 or 4: {x.ndim}")
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.to(self.device)
        return x

    def get_mask_img(self):
        if self.mask_img is None:
            self.mask_img = cv2.imread(MASK_TEMPLATES, cv2.IMREAD_COLOR)
        return self.mask_img

    def crop_face(self, img_rgb, crop_factor):
        crop_region = self.detect_face(img_rgb, crop_factor)
        face_region, is_changed = self.calc_face_region(crop_region, get_rgb_size(img_rgb))
        face_img = rgb_crop(img_rgb, face_region)
        if is_changed:
            face_img = self.expand_img(face_img, crop_region)
        return face_img

    def prepare_source(self, source_image, crop_factor, is_video=False, tracking=False, pre_cropped=False):
        # source_image_np = (source_image * 255).byte().numpy()
        # img_rgb = source_image_np[0]
        # print("Prepare source...")
        if isinstance(source_image, str):
            source_image = image_path_to_array(source_image)

        if len(source_image.shape) <= 3:
            source_image = source_image[np.newaxis, ...]

        psi_list = []
        for img_rgb in source_image:
            if tracking or len(psi_list) == 0:
                if pre_cropped:
                    crop_region = [0, 0, img_rgb.shape[1], img_rgb.shape[0]]
                else:
                    crop_region = self.detect_face(img_rgb, crop_factor)
                face_region, is_changed = self.calc_face_region(crop_region, get_rgb_size(img_rgb))

                s_x = (face_region[2] - face_region[0]) / 512.0
                s_y = (face_region[3] - face_region[1]) / 512.0
                crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], s_x, s_y)
                mask_ori = cv2.warpAffine(self.get_mask_img(), crop_trans_m, get_rgb_size(img_rgb), cv2.INTER_LINEAR)
                mask_ori = mask_ori.astype(np.float32) / 255.0

                if is_changed:
                    s = (crop_region[2] - crop_region[0]) / 512.0
                    crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], s, s)

            face_img = rgb_crop(img_rgb, face_region)
            if is_changed:
                face_img = self.expand_img(face_img, crop_region)
            i_s = self.prepare_src_image(face_img)
            x_s_info = self.pipeline.get_kp_info(i_s)  # ['exp', 'yaw', 'pitch', 'roll', 't', 'scale']
            f_s_user = self.pipeline.extract_feature_3d(i_s)
            x_s_user = self.pipeline.transform_keypoint(x_s_info)
            psi = PreparedSrcImg(img_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori)
            if is_video == False:
                return psi
            psi_list.append(psi)

        return psi_list

    def prepare_driving_video(self, face_images):
        print("Prepare driving video...")
        f_img_np = (face_images * 255).byte().numpy()

        out_list = []
        for f_img in f_img_np:
            i_d = self.prepare_src_image(f_img)
            d_info = self.pipeline.get_kp_info(i_d)
            out_list.append(d_info)

        return out_list

    @staticmethod
    def calc_fe(
        x_d_new, eyes, eyebrow, wink, pupil_x, pupil_y, mouth, eee, woo, smile, rotate_pitch, rotate_yaw, rotate_roll
    ):
        x_d_new[0, 20, 1] += smile * -0.01
        x_d_new[0, 14, 1] += smile * -0.02
        x_d_new[0, 17, 1] += smile * 0.0065
        x_d_new[0, 17, 2] += smile * 0.003
        x_d_new[0, 13, 1] += smile * -0.00275
        x_d_new[0, 16, 1] += smile * -0.00275
        x_d_new[0, 3, 1] += smile * -0.0035
        x_d_new[0, 7, 1] += smile * -0.0035

        x_d_new[0, 19, 1] += mouth * 0.001
        x_d_new[0, 19, 2] += mouth * 0.0001
        x_d_new[0, 17, 1] += mouth * -0.0001
        rotate_pitch -= mouth * 0.05

        x_d_new[0, 20, 2] += eee * -0.001
        x_d_new[0, 20, 1] += eee * -0.001
        # x_d_new[0, 19, 1] += eee * 0.0006
        x_d_new[0, 14, 1] += eee * -0.001

        x_d_new[0, 14, 1] += woo * 0.001
        x_d_new[0, 3, 1] += woo * -0.0005
        x_d_new[0, 7, 1] += woo * -0.0005
        x_d_new[0, 17, 2] += woo * -0.0005

        x_d_new[0, 11, 1] += wink * 0.001
        x_d_new[0, 13, 1] += wink * -0.0003
        x_d_new[0, 17, 0] += wink * 0.0003
        x_d_new[0, 17, 1] += wink * 0.0003
        x_d_new[0, 3, 1] += wink * -0.0003
        rotate_roll -= wink * 0.1
        rotate_yaw -= wink * 0.1

        if 0 < pupil_x:
            x_d_new[0, 11, 0] += pupil_x * 0.0007
            x_d_new[0, 15, 0] += pupil_x * 0.001
        else:
            x_d_new[0, 11, 0] += pupil_x * 0.001
            x_d_new[0, 15, 0] += pupil_x * 0.0007

        x_d_new[0, 11, 1] += pupil_y * -0.001
        x_d_new[0, 15, 1] += pupil_y * -0.001
        eyes -= pupil_y / 2.0

        x_d_new[0, 11, 1] += eyes * -0.001
        x_d_new[0, 13, 1] += eyes * 0.0003
        x_d_new[0, 15, 1] += eyes * -0.001
        x_d_new[0, 16, 1] += eyes * 0.0003
        x_d_new[0, 1, 1] += eyes * -0.00025
        x_d_new[0, 2, 1] += eyes * 0.00025

        if 0 < eyebrow:
            x_d_new[0, 1, 1] += eyebrow * 0.001
            x_d_new[0, 2, 1] += eyebrow * -0.001
        else:
            x_d_new[0, 1, 0] += eyebrow * -0.001
            x_d_new[0, 2, 0] += eyebrow * 0.001
            x_d_new[0, 1, 1] += eyebrow * 0.0003
            x_d_new[0, 2, 1] += eyebrow * -0.0003

        return torch.Tensor([rotate_pitch, rotate_yaw, rotate_roll])


class ExpressionSet:
    def __init__(self, erst=None, es=None, gpu_id: int = 0):
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        if es is not None:
            self.e = copy.deepcopy(es.e).to(device)
            self.r = copy.deepcopy(es.r).to(device)
            self.s = copy.deepcopy(es.s)
            self.t = copy.deepcopy(es.t)
        elif erst is not None:
            self.e = erst[0].to(device)
            self.r = erst[1].to(device)
            self.s = erst[2]
            self.t = erst[3]
        else:
            self.e = torch.zeros((1, 21, 3), device=device)
            self.r = torch.zeros(3, device=device)
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

    @staticmethod
    def get_device(gpu_id: int = 0):
        if torch.cuda.is_available():
            return f"cuda:{gpu_id}"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time - start_time))
        return result

    return wrapper_fn


def save_exp_data(file_name: str, save_exp: ExpressionSet = None):
    if save_exp is None or not file_name:
        return file_name

    with open(os.path.join(EXP_OUTPUT_DIR, file_name + ".exp"), "wb") as f:
        dill.dump(save_exp, f)

    return file_name


def load_exp_data(self, file_name, ratio):
    file_list = [os.path.splitext(file)[0] for file in os.listdir(EXP_OUTPUT_DIR) if file.endswith(".exp")]
    with open(os.path.join(EXP_OUTPUT_DIR, file_name + ".exp"), "rb") as f:
        es = dill.load(f)
    es.mul(ratio)
    return es


def handle_exp_data(code1, value1, code2, value2, code3, value3, code4, value4, code5, value5, add_exp=None):
    if add_exp is None:
        es = ExpressionSet()
    else:
        es = ExpressionSet(es=add_exp)

    codes = [code1, code2, code3, code4, code5]
    values = [value1, value2, value3, value4, value5]
    for i in range(5):
        idx = int(codes[i] / 10)
        r = codes[i] % 10
        es.e[0, idx, r] += values[i] * 0.001

    return es


def print_exp_data(cut_noise, exp=None):
    if exp is None:
        return exp

    cuted_list = []
    e = exp.exp * 1000
    for idx in range(21):
        for r in range(3):
            a = abs(e[0, idx, r])
            if cut_noise < a:
                cuted_list.append((a, e[0, idx, r], idx * 10 + r))

    sorted_list = sorted(cuted_list, reverse=True, key=lambda item: item[0])
    print(f"sorted_list: {[[item[2], round(float(item[1]), 1)] for item in sorted_list]}")
    return exp


class Command:
    def __init__(self, es: ExpressionSet, change, keep):
        self.es = es
        self.change = change
        self.keep = keep
