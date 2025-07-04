import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class SegFormerFaceParser:
    def __init__(self, device=None):
        self.image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

    def _preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        return self.image_processor(images=images, return_tensors="pt").pixel_values.to(self.device)

    def parse(self, images: list[np.ndarray], input_type: str = "channel_first") -> np.ndarray:
        if isinstance(images, list):
            assert len(images[0].shape) in [3, 4], "Only one image is supported"
        else:
            assert len(images.shape) in [3, 4], "Only one image is supported"
        if input_type == "channel_first":
            size = images[0].shape[1:] if len(images[0].shape) == 3 else images[2:].shape
        elif input_type == "channel_last":
            size = images[0].shape[:2] if len(images[0].shape) == 3 else images[1:3].shape
        else:
            raise ValueError(f"Invalid input type: {input_type}")
        image_tensors = self._preprocess(images)
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
        return labels.cpu().numpy().astype(np.uint8)


if __name__ == "__main__":
    import fastcore.all as fc
    import torchvision
    from PIL import Image
    from torchvision.transforms.v2 import ColorJitter

    # from src.face_parsing.face_parser import segmentation_mask

    # folders = fc.L(fc.Path("/data/prakash_lipsync/video_hallo/iv_recording_v2/").glob("*"))
    # folder = folders[np.random.randint(0, len(folders))]
    # bins = fc.L(folder.glob("*"))
    # imgs = bins[np.random.randint(0, len(bins))].glob("*.png")
    # imgs = fc.L(imgs)
    imgs = fc.L(fc.Path("nbs/assets/storage2_gt").glob("*.png"))
    # image = Image.open(imgs[0])
    img = imgs[np.random.randint(0, len(imgs))]
    image = torchvision.io.read_image(img)
    image2 = Image.open(img)
    tf1 = ColorJitter(brightness=0.5, contrast=0.5, saturation=None, hue=None)
    tf2 = ColorJitter(brightness=None, contrast=None, saturation=0.5, hue=0.5)
    image = tf1(image)
    image = tf2(image)
    parser = SegFormerFaceParser()
    image = torch.stack([image, image]) / 255.0
    res = parser.parse(image * 255, input_type="channel_first")
    # res2 = parser.parse([np.asarray(image2)], input_type="channel_last")
    res, res2 = res[0], res[1]

    res = np.repeat(res[None], 3, axis=0)
    res2 = np.repeat(res2[None], 3, axis=0)
    # permute to (h, w, c)
    res = res.transpose(1, 2, 0)
    res2 = res2.transpose(1, 2, 0)
    res = res.copy()
    res[res > 0] = 1
    res2 = res2.copy()
    res2[res2 > 0] = 1
    grid = Image.fromarray(res)
    grid2 = Image.fromarray(res2 * 255)

    # get background
    bg_img = np.asarray(image2)
    bg_img = bg_img * (res == 0)
    bg_img = Image.fromarray(bg_img)

    bg_img2 = np.asarray(image2)
    bg_img2 = bg_img2 * (res2 == 0)
    bg_img2 = Image.fromarray(bg_img2)

    x = Image.new("RGB", (image2.width * 3, image2.height), (255, 255, 255))
    x.paste(image2, (0, 0))
    x.paste(bg_img, (image2.width, 0))
    x.paste(bg_img2, (image2.width * 2, 0))
    x.save("temp/face_parse.png")
