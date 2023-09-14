"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

import argparse
import os
import sys

import mindspore
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore.dataset import CelebADataset
from mindspore.dataset.vision import transforms, Inter

from img_utils import to_image
from models import GeneratorRRDB, Discriminator, FeatureExtractor

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.set_train(False)

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_content = nn.L1Loss()
criterion_pixel = nn.L1Loss()

if opt.epoch != 0:
    # Load pretrained models
    mindspore.load_checkpoint(f'saved_models/generator_{opt.epoch}.ckpt', generator)
    mindspore.load_checkpoint(f'saved_models/discriminator{opt.epoch}.ckpt', discriminator)

# Optimizers
optimizer_G = nn.optim.Adam(generator.trainable_params(), opt.lr, opt.b1, opt.b2)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), opt.lr, opt.b1, opt.b2)

def preprocess(_imgs):
    """Dataset preprocess func"""
    _imgs = transforms.Resize((opt.hr_height, opt.hr_width), Inter.BICUBIC)(_imgs)
    _imgs_lr = transforms.Resize((opt.hr_height // 4, opt.hr_width // 4), Inter.BICUBIC)(_imgs)
    _imgs = transforms.ToTensor()(_imgs)
    _imgs = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False)(_imgs)
    _imgs_lr = transforms.ToTensor()(_imgs_lr)
    _imgs_lr = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False)(_imgs_lr)

    return _imgs, _imgs_lr


dataset = CelebADataset(
    dataset_dir="../../data/CelebA",
    shuffle=True,
    decode=True
).map(operations=[preprocess], input_columns=["image"], output_columns=["image_hr", "image_lr"]).batch(opt.batch_size)


def g_forward(_imgs_hr, _imgs_lr, _valid):
    """Generator forward function"""
    # Generate a high resolution image from low resolution input
    _gen_hr = generator(_imgs_lr)
    # Measure pixel-wise loss against ground truth
    _pixel_loss = criterion_pixel(_gen_hr, _imgs_hr)

    # Extract validity predictions from discriminator
    pred_real = ops.stop_gradient(discriminator(_imgs_hr))
    pred_fake = discriminator(_gen_hr)

    # Adversarial loss (relativistic average GAN)
    _GAN_loss = criterion_GAN(pred_fake - pred_real.mean(0, keep_dims=True), _valid)

    # Content loss
    gen_features = feature_extractor(_gen_hr)
    real_features = ops.stop_gradient(feature_extractor(_imgs_hr))
    _content_loss = criterion_content(gen_features, real_features)

    # Total generator loss
    _g_loss = _content_loss + opt.lambda_adv * _GAN_loss + opt.lambda_pixel * loss_pixel

    return _g_loss, _content_loss, _GAN_loss, _pixel_loss, _gen_hr


def g_forward_warmup(_imgs_hr, _imgs_lr):
    """Generator warmup forward func"""
    # Generate a high resolution image from low resolution input
    _gen_hr = generator(_imgs_lr)
    # Measure pixel-wise loss against ground truth
    _loss_pixel = criterion_pixel(_gen_hr, _imgs_hr)
    return _loss_pixel


def d_forward(_imgs_hr, _gen_hr, _valid, _fake):
    """Discriminator forward function"""
    pred_real = discriminator(_imgs_hr)
    pred_fake = discriminator(_gen_hr)

    # Adversarial loss for real and fake images (relativistic average GAN)
    loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keep_dims=True), _valid)
    loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keep_dims=True), _fake)

    # Total loss
    _d_loss = (loss_real + loss_fake) / 2
    return _d_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)
grad_g_warmup = ops.value_and_grad(g_forward_warmup, None, optimizer_G.parameters, has_aux=False)
# ----------
#  Training
# ----------

saved_samples = {}

for epoch in range(opt.n_epochs):
    # Model inputs
    for i, batch in enumerate(dataset.create_dict_iterator()):

        batches_done = epoch * dataset.get_dataset_size() + i

        imgs_hr = batch["image_hr"]
        imgs_lr = batch["image_lr"]

        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((imgs_lr.shape[0], *discriminator.output_shape)))
        fake = ops.stop_gradient(ops.zeros((imgs_lr.shape[0], *discriminator.output_shape)))

        # ------------------
        #  Train Generators
        # ------------------

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            (loss_pixel), g_grads = grad_g_warmup(imgs_hr, imgs_lr)
            optimizer_G(g_grads)
            print(
                f'[Epoch {epoch}/{opt.n_epochs}] '
                f'[Batch {i}/{dataset.get_dataset_size()}] '
                f'[G pixel: {loss_pixel.asnumpy().item()}]'
            )
            continue

        (g_loss, content_loss, GAN_loss, pixel_loss, gen_hr), g_grads = grad_g(imgs_hr, imgs_lr, valid)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        (d_loss), d_grads = grad_d(imgs_hr, ops.stop_gradient(gen_hr),
                                   valid, fake)
        optimizer_D(d_grads)

        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write(
            f'\r[Epoch {epoch}/{opt.n_epochs}] '
            f'[Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] '
            f'[G loss: {g_loss.asnumpy().item():.4f}, '
            f'content: {content_loss.asnumpy().item():.4f}, '
            f'adv: {GAN_loss.asnumpy().item():.4f}, '
            f'pixel:{pixel_loss.asnumpy().item():.4f}]'
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = ops.interpolate(imgs_lr, scale_factor=4.0, recompute_scale_factor=True)
            img_grid = denormalize(ops.cat((imgs_lr, gen_hr), -1))
            to_image(img_grid, f'images/training/{batches_done}.png')

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        mindspore.save_checkpoint(generator, f'saved_models/{opt.dataset_name}/generator_{epoch}.ckpt')
        mindspore.save_checkpoint(discriminator, f'saved_models/{opt.dataset_name}/discriminator_{epoch}.ckpt')
