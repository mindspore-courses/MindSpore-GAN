"""CCGAN Model"""

import argparse
import os

import mindspore
import numpy as np
from mindspore import Tensor, ops
from mindspore import nn
from mindspore.dataset import CelebADataset
from mindspore.dataset.vision import transforms, Inter

from img_utils import to_image
from models import Generator, Discriminator

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

input_shape = (opt.channels, opt.img_size, opt.img_size)

# Loss function
adversarial_loss = nn.MSELoss()

# Initialize generator and discriminator
generator = Generator(input_shape)
discriminator = Discriminator(input_shape)

def preprocess(_imgs):
    """Dataset preprocess func"""
    _imgs = transforms.Resize((opt.img_size, opt.img_size), Inter.BICUBIC)(_imgs)
    _imgs_lr = transforms.Resize((opt.img_size // 4, opt.img_size // 4), Inter.BICUBIC)(_imgs)
    _imgs = transforms.ToTensor()(_imgs)
    _imgs = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False)(_imgs)
    _imgs_lr = transforms.ToTensor()(_imgs_lr)
    _imgs_lr = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False)(_imgs_lr)

    return _imgs, _imgs_lr


dataset = CelebADataset(
    dataset_dir="../../data/CelebA",
    shuffle=True,
    decode=True
).map(operations=[preprocess], input_columns=["image"], output_columns=["image", "image_lr"]).batch(opt.batch_size)

# Optimizers
optimizer_G = nn.optim.Adam(generator.trainable_params(), opt.lr, opt.b1, opt.b2)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), opt.lr, opt.b1, opt.b2)


def apply_random_mask(_imgs):
    """Add random masks to images."""
    idx = np.random.randint(0, opt.img_size - opt.mask_size, (_imgs.shape[0], 2))

    _masked_imgs = _imgs.copy().asnumpy()
    for j, (y1, x1) in enumerate(idx):
        y2, x2 = y1 + opt.mask_size, x1 + opt.mask_size
        _masked_imgs[j, :, y1:y2, x1:x2] = -1

    return Tensor(_masked_imgs)


def save_sample(_saved_samples):
    """Save samples."""
    # Generate inpainted image
    _gen_imgs = generator(_saved_samples["masked"], _saved_samples["lowres"])
    # Save sample
    sample = ops.cat((_saved_samples["masked"], _gen_imgs, _saved_samples["imgs"]), -2)
    to_image(sample, os.path.join(f'images/{opt.dataset_name}', F'{batches_done}.png'))


def g_forward(_masked_imgs, _imgs_lr, _valid):
    """Generator forward function"""
    # Generate a batch of images
    _gen_imgs = generator(_masked_imgs, _imgs_lr)

    # Loss measures generator's ability to fool the discriminator
    _g_loss = adversarial_loss(discriminator(_gen_imgs), _valid)
    return _g_loss, _gen_imgs


def d_forward(_real_imgs, _gen_imgs, _valid, _fake):
    """Discriminator forward function"""
    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(_real_imgs), _valid)
    fake_loss = adversarial_loss(discriminator(_gen_imgs), _fake)
    _d_loss = 0.5 * (real_loss + fake_loss)
    return _d_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)

# ----------
#  Training
# ----------

saved_samples = {}

for epoch in range(opt.n_epochs):
    # Model inputs
    for i, batch in enumerate(dataset.create_dict_iterator()):
        imgs = batch["image"]
        imgs_lr = batch["image_lr"]

        masked_imgs = apply_random_mask(imgs)

        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((imgs.shape[0], *discriminator.output_shape)))
        fake = ops.stop_gradient(ops.zeros((imgs.shape[0], *discriminator.output_shape)))

        real_imgs = Tensor(imgs)
        imgs_lr = Tensor(imgs_lr)
        masked_imgs = Tensor(masked_imgs)

        # -----------------
        #  Train Generator
        # -----------------

        (g_loss, gen_imgs), g_grads = grad_g(masked_imgs, imgs_lr, valid)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        (d_loss), d_grads = grad_d(real_imgs,
                                   ops.stop_gradient(gen_imgs), valid, fake)
        optimizer_D(d_grads)

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f} [G loss: {g_loss.asnumpy().item():.4f}]'
        )

        # Save first ten samples
        if not saved_samples:
            saved_samples["imgs"] = real_imgs[:1].copy()
            saved_samples["masked"] = masked_imgs[:1].copy()
            saved_samples["lowres"] = imgs_lr[:1].copy()
        elif saved_samples["imgs"].shape[0] < 10:
            saved_samples["imgs"] = ops.cat((saved_samples["imgs"], real_imgs[:1]), 0)
            saved_samples["masked"] = ops.cat((saved_samples["masked"], masked_imgs[:1]), 0)
            saved_samples["lowres"] = ops.cat((saved_samples["lowres"], imgs_lr[:1]), 0)

        batches_done = epoch * dataset.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            save_sample(saved_samples)
        if batches_done % 5000 == 0:
            mindspore.save_checkpoint(generator,f'./g-{batches_done}.ckpt')
            mindspore.save_checkpoint(discriminator, f'./d-{batches_done}.ckpt')
