"""
Inpainting using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 context_encoder.py'
"""

import argparse
import os

import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore.dataset import CelebADataset
from mindspore.dataset.vision import transforms, Inter

from img_utils import to_image
from models import Generator, Discriminator

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
patch = (1, patch_h, patch_w)

adversarial_loss = nn.MSELoss()
pixelwise_loss = nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(channels=opt.channels)
discriminator = Discriminator(channels=opt.channels)

# Optimizers
optimizer_G = nn.optim.Adam(generator.trainable_params(), opt.lr, opt.b1, opt.b2)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), opt.lr, opt.b1, opt.b2)


def apply_random_mask(img, img_size, mask_size):
    """Randomly masks image"""
    y1, x1 = np.random.randint(0, img_size - mask_size, 2)
    y2, x2 = y1 + mask_size, x1 + mask_size
    _masked_part = img[:, y1:y2, x1:x2]
    _masked_img = img.copy()
    _masked_img[:, y1:y2, x1:x2] = 1

    return _masked_img, _masked_part


def apply_center_mask(img, img_size, mask_size):
    """Mask center part of image"""
    # Get upper-left pixel coordinate
    k = (img_size - mask_size) // 2
    masked_img = img.copy()
    masked_img[:, k: k + mask_size, k: k + mask_size] = 1

    return masked_img, k


def train_preprocess(_imgs):
    """Dataset preprocess func"""
    _imgs = transforms.Resize((opt.img_size, opt.img_size), Inter.BICUBIC)(_imgs)
    _imgs = transforms.ToTensor()(_imgs)
    _imgs = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False)(_imgs)
    # For training data perform random mask
    _masked_img, _aux = apply_random_mask(_imgs, 128, 64)
    return _imgs, _masked_img, _aux


def test_preprocess(_imgs):
    """Dataset preprocess func"""
    _imgs = transforms.Resize((opt.img_size, opt.img_size), Inter.BICUBIC)(_imgs)
    _imgs = transforms.ToTensor()(_imgs)
    _imgs = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False)(_imgs)
    # For test data mask the center of the image
    _masked_img, _aux = apply_center_mask(_imgs, 128, 64)
    return _imgs, _masked_img, _aux


train_dataset = CelebADataset(
    dataset_dir="../../data/CelebA",
    shuffle=True,
    decode=True
).map(operations=[train_preprocess], input_columns=["image"], output_columns=["image_hr", "masked_image", "aux"]).batch(
    opt.batch_size)

test_dataset = CelebADataset(
    dataset_dir="../../data/CelebA",
    shuffle=True,
    decode=True
).map(operations=[test_preprocess], input_columns=["image"], output_columns=["image_hr", "masked_image", "aux"]).batch(
    12)


def save_sample(_batches_done):
    """Save sample images"""
    samples, masked_samples, j, _ = next(test_dataset.create_tuple_iterator())
    j = j[0].asnumpy().item()  # Upper-left coordinate of mask
    # Generate inpainted image
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.copy()
    filled_samples[:, :, j: j + opt.mask_size, j: j + opt.mask_size] = gen_mask
    # Save sample
    sample = ops.cat((masked_samples, filled_samples, samples), -2)
    to_image(sample, f'images/{_batches_done}.png')


def g_forward(_masked_imgs, _masked_parts, _valid):
    """GeneratorUNet warmup forward func"""
    # Generate a batch of images
    _gen_parts = generator(_masked_imgs)

    # Adversarial and pixelwise loss
    _g_adv = adversarial_loss(discriminator(_gen_parts), _valid)
    _g_pixel = pixelwise_loss(_gen_parts, _masked_parts)
    # Total loss
    _g_loss = 0.001 * _g_adv + 0.999 * _g_pixel
    return _g_loss, _g_adv, _g_pixel, _gen_parts


def d_forward(_masked_parts, _gen_parts, _valid, _fake):
    """Discriminator forward function"""
    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(_masked_parts), _valid)
    fake_loss = adversarial_loss(discriminator(_gen_parts), _fake)
    _d_loss = 0.5 * (real_loss + fake_loss)
    return _d_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)

for epoch in range(opt.n_epochs):
    # Model inputs
    for i, (imgs, masked_imgs, masked_parts, _) in enumerate(train_dataset.create_tuple_iterator()):
        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((imgs.shape[0], *patch)))
        fake = ops.stop_gradient(ops.zeros((imgs.shape[0], *patch)))

        # -----------------
        #  Train GeneratorUNet
        # -----------------
        (g_loss, g_adv, g_pixel, gen_parts), g_grads = grad_g(masked_imgs, masked_parts, valid)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        (d_loss), d_grads = grad_d(masked_parts, ops.stop_gradient(gen_parts), valid, fake)
        optimizer_D(d_grads)

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] '
            f'[Batch {i}/{train_dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] '
            f'[G adv: {g_adv.asnumpy().item():.4f}, '
            f'pixel: {g_pixel.asnumpy().item():.4f}]'
        )

        # Generate sample at sample interval
        batches_done = epoch * train_dataset.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)
