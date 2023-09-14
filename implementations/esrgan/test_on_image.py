"""Test ESRGAN"""
import argparse
import os

import mindspore
from mindspore import Tensor, ops
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import transforms

from PIL import Image

from models import GeneratorRRDB
from esrgan import denormalize, mean, std
from img_utils import to_image

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to image")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)

os.makedirs("images/outputs", exist_ok=True)
# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks)
mindspore.load_checkpoint(opt.checkpoint_model, generator)
generator.set_train(False)

transform = Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std, is_hwc=False)]
)

# Prepare input
image_tensor = Tensor(transform(Image.open(opt.image_path))).unsqueeze(0)

sr_image = denormalize(ops.stop_gradient(generator(image_tensor)))

fn = opt.image_path.split("/")[-1]
to_image(sr_image, f'images/outputs/sr-{fn}')
