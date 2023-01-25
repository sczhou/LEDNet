# Modified by Shangchen Zhou from: https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py
import os

import cv2
import torch
from cog import BasePredictor, Input, Path
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor, imwrite, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY
from inference_lednet import check_image_size

pretrain_model_url = {
    "lednet": "https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet.pth",
    "lednet_retrain": "https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet_retrain_500000.pth",
}

POTENTIAL_MODELS = list(pretrain_model_url.keys())
DOWN_FACTOR = 8  # check_image_size
OUT_PATH = "./results"


class LEDNetPredictor(BasePredictor):
    """
    Predictor wrapper around LEDNet
    """

    def setup(self):
        """
        One-time setup method to load and prep model for efficient prediction.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}

        for model in POTENTIAL_MODELS:
            net = ARCH_REGISTRY.get("LEDNet")(
                channels=[32, 64, 128, 128], connection=False
            ).to(self.device)

            ckpt_path = load_file_from_url(
                url=pretrain_model_url[model],
                model_dir="/weights",
                progress=True,
                file_name=None,
            )
            checkpoint = torch.load(ckpt_path, map_location=self.device)["params"]
            net.load_state_dict(checkpoint)
            net.eval()
            self.models[model] = net

    def predict(
        self,
        model: str = Input(
            default="lednet",
            description="pretrained model to use for inference",
            choices=POTENTIAL_MODELS,
        ),
        image: Path = Input(description="Input image"),
    ) -> Path:
        """
        Runs inference with selected model on input image.
        """
        net = self.models[model]

        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        # prepare data
        img_t = img2tensor(img / 255.0, bgr2rgb=True, float32=True)

        # without [-1,1] normalization in lednet model (paper version)
        if not model == "lednet":
            normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        img_t = img_t.unsqueeze(0).to(self.device)

        # lednet inference
        with torch.no_grad():
            # check_image_size
            H, W = img_t.shape[2:]
            img_t = check_image_size(img_t, DOWN_FACTOR)
            output_t = net(img_t)
            output_t = output_t[:, :, :H, :W]

            if model == "lednet":
                output = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1))
            else:
                output = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))

        del output_t
        torch.cuda.empty_cache()

        output = output.astype("uint8")
        # save restored img
        save_restore_path = os.path.join(OUT_PATH, "out.jpg")
        imwrite(output, save_restore_path)

        return Path(save_restore_path)
