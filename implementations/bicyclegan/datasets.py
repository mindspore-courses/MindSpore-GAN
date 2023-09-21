"""Dataset Preprocess"""

import glob
import os

import numpy as np
from PIL import Image
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import transforms, Inter


class Edges2ShoesDataset():
    """Edges2Shoes Dataset"""

    def __init__(self, root, input_shape, mode="train"):
        self.transform = Compose(
            [
                transforms.Resize(input_shape[-2:], Inter.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], is_hwc=False)
            ]
        )

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return img_A[0], img_B[0]

    def __len__(self):
        return len(self.files)
