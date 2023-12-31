"""图像处理"""
import math
import pathlib
from typing import BinaryIO, Union, Optional, Tuple
import mindspore
from mindspore import ops
import mindspore.numpy as mnp
from PIL import Image



def make_grid(
        tensor,
        nrow: int = 1,
        padding: int = 2,
        normalize: bool = True,
        value_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: float = 0.0):
    """grid"""
    if isinstance(tensor, list):
        tensor = mnp.stack(tensor, axis=0)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = mnp.concatenate((tensor, tensor, tensor), 0)
        tensor = tensor.expand_dims(0)

    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = mnp.concatenate((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor_list = []
        if value_range is not None:
            assert isinstance(
                value_range, tuple
            ), "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img = ops.clip_by_value(img, low, high)
            img = (img - low) / (max(high - low, 1e-5))
            return img

        def norm_range(input_tensor, value_range):
            if value_range is not None:
                return norm_ip(input_tensor, value_range[0], value_range[1])
            return norm_ip(input_tensor, float(input_tensor.min()), float(input_tensor.max()))

        if scale_each is True:
            for input_t in tensor:  # loop over mini-batch dimension
                tensor_list.append(norm_range(input_t, value_range))
        else:
            tensor_list = norm_range(tensor, value_range)

        if isinstance(tensor_list, mindspore.Tensor):
            tensor = tensor_list
        else:
            tensor = mnp.concatenate(tensor_list)

    assert isinstance(tensor, mindspore.Tensor)
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = mnp.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding: (y + 1) * height, x * width + padding: (x + 1) * width] = tensor[k]
            k = k + 1
    return grid


def to_image(tensor,
             file: Union[str, pathlib.Path, BinaryIO],
             _format=None,
             **kwargs):
    """保存图片"""
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid * 255 + 0.5
    ndarr = ops.clip_by_value(ndarr, 0, 255).transpose(1, 2, 0).astype(mindspore.uint8).asnumpy()
    img = Image.fromarray(ndarr)
    img.save(file, format=_format)
