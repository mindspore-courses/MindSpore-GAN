"""BicycleGAN Model"""

import argparse
import datetime
import os
import sys
import time

import mindspore
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import ops

from datasets import Edges2ShoesDataset
from img_utils import to_image
from models import Generator, MultiDiscriminator, Encoder

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
opt = parser.parse_args()
print(opt)

os.makedirs(f'images/{opt.dataset_name}', exist_ok=True)
os.makedirs(f'saved_models/{opt.dataset_name}', exist_ok=True)

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Loss functions
mae_loss = nn.L1Loss()

# Initialize generator, encoder and discriminators
generator = Generator(opt.latent_dim, input_shape)
encoder = Encoder(opt.latent_dim)
D_VAE = MultiDiscriminator(input_shape)
D_LR = MultiDiscriminator(input_shape)

generator.update_parameters_name("generator")
encoder.update_parameters_name("encoder")
D_VAE.update_parameters_name("D_VAE")
D_LR.update_parameters_name("D_LR")

if opt.epoch != 0:
    # Load pretrained models
    # Load pretrained models
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/generator_{opt.epoch}.ckpt', generator)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/encoder_{opt.epoch}.ckpt', encoder)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/D_VAE_{opt.epoch}.ckpt', D_VAE)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/D_LR_{opt.epoch}.ckpt', D_LR)

# Optimizers
optimizer_E = nn.optim.Adam(encoder.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_G = nn.optim.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D_VAE = nn.optim.Adam(D_VAE.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D_LR = nn.optim.Adam(D_LR.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)

optimizer_E.update_parameters_name("optimizer_E")
optimizer_G.update_parameters_name("optimizer_G")
optimizer_D_VAE.update_parameters_name("optimizer_D_VAE")
optimizer_D_LR.update_parameters_name("optimizer_D_LR")

dataset = mindspore.dataset.GeneratorDataset(
    source=Edges2ShoesDataset(
        root="../../data/edges2shoes",
        input_shape=input_shape
    ),
    column_names=["A", "B"],
    shuffle=True
).batch(opt.batch_size)

val_dataset = mindspore.dataset.GeneratorDataset(
    source=Edges2ShoesDataset(
        root="../../data/edges2shoes",
        input_shape=input_shape,
        mode='val'
    ),
    column_names=["A", "B"],
    shuffle=True
).batch(8)


def sample_image(batches):
    """Saves a generated sample from the validation set"""
    generator.set_train(False)
    imgs = next(val_dataset.create_tuple_iterator())
    img_samples = None
    for img_A, _ in zip(imgs[0], imgs[1]):
        # Repeat input image by number of desired columns
        _real_A = img_A.view(1, *img_A.shape).tile((opt.latent_dim, 1, 1, 1))
        # Sample latent representations
        sampled_z = ops.randn(opt.latent_dim, opt.latent_dim, dtype=mstype.float32)
        # Generate samples
        fake_B = generator(_real_A, sampled_z)
        # Concatenate samples horisontally
        fake_B = ops.cat(list(fake_B), -1)
        img_sample = ops.cat((img_A, fake_B), -1)
        img_sample = img_sample.view(1, *img_sample.shape)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else ops.cat((img_samples, img_sample), -2)
    to_image(img_samples, os.path.join(f'images/{opt.dataset_name}', f'{batches}.png'))
    generator.set_train()


def reparameterization(_mu, _logvar):
    """Reparameterization function"""
    std = ops.exp(_logvar / 2)
    sampled_z = ops.randn(_mu.shape[0], opt.latent_dim, dtype=mstype.float32)
    _z = sampled_z * std + _mu
    return _z


def ge_forward(_real_A, _real_B, _valid):
    """Encoder forward func"""
    # Produce output using encoding of B (cVAE-GAN)
    mu, logvar = encoder(_real_B)
    encoded_z = reparameterization(mu, logvar)
    fake_B = generator(_real_A, encoded_z)

    # Pixelwise loss of translated image by VAE
    _loss_pixel = mae_loss(fake_B, _real_B)
    # Kullback-Leibler divergence of encoded B
    _loss_kl = 0.5 * ops.sum(ops.exp(logvar) + mu ** 2 - logvar - 1)
    # Adversarial loss
    loss_VAE_GAN = D_VAE.compute_loss(fake_B, _valid)

    # ---------
    # cLR-GAN
    # ---------

    # Produce output using sampled z (cLR-GAN)
    sampled_z = ops.randn(_real_A.shape[0], opt.latent_dim, dtype=mstype.float32)
    _fake_B = generator(_real_A, sampled_z)
    # cLR Loss: Adversarial loss
    loss_LR_GAN = D_LR.compute_loss(_fake_B, _valid)

    # ----------------------------------
    # Total Loss (Generator + Encoder)
    # ----------------------------------

    _loss_GE = loss_VAE_GAN + loss_LR_GAN + opt.lambda_pixel * _loss_pixel + opt.lambda_kl * _loss_kl
    return _loss_GE, _loss_pixel, _loss_kl, fake_B, _fake_B


def g_forward(_real_A):
    """Generator warmup forward func"""
    # Produce output using sampled z (cLR-GAN)
    sampled_z = ops.randn(_real_A.shape[0], opt.latent_dim, dtype=mstype.float32)
    _fake_B = generator(_real_A, sampled_z)

    # Latent L1 loss
    _mu, _ = encoder(_fake_B)
    _loss_latent = opt.lambda_latent * mae_loss(_mu, sampled_z)

    return _loss_latent


def d_vae_forward(_real_B, _fake_B, _valid, _fake):
    """Discriminator forward function"""

    _loss_D_VAE = D_VAE.compute_loss(_real_B, _valid) + D_VAE.compute_loss(_fake_B, _fake)

    return _loss_D_VAE


def d_lr_forward(_real_B, _fake_B, _valid, _fake):
    """Discriminator forward function"""

    _loss_D_LR = D_LR.compute_loss(_real_B, _valid) + D_LR.compute_loss(_fake_B, _fake)

    return _loss_D_LR


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=False)
grad_d_vae = ops.value_and_grad(d_vae_forward, None, optimizer_D_VAE.parameters, has_aux=False)
grad_ge = ops.value_and_grad(ge_forward, None, optimizer_E.parameters, has_aux=True)
grad_d_lr = ops.value_and_grad(d_lr_forward, None, optimizer_D_LR.parameters, has_aux=False)

# ----------
#  Training
# ----------

generator.set_train()
encoder.set_train()
D_LR.set_train()
D_VAE.set_train()

# Adversarial loss
valid = 1
fake = 0

prev_time = time.time()
for epoch in range(opt.n_epochs):
    # Model inputs
    for i, (real_A, real_B) in enumerate(dataset.create_tuple_iterator()):

        # -------------------------------
        #  Train Generator and Encoder
        # -------------------------------

        (loss_GE, loss_pixel, loss_kl, fake_B1, fake_B2), ge_grads = grad_ge(real_A, real_B, valid)
        optimizer_E(ge_grads)

        (loss_latent), g_grads = grad_g(real_A)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        (loss_D_VAE), d_vae_grads = grad_d_vae(real_B, ops.stop_gradient(fake_B1), valid, fake)
        optimizer_D_VAE(d_vae_grads)

        (loss_D_LR), d_lr_grads = grad_d_lr(real_B, ops.stop_gradient(fake_B2), valid, fake)
        optimizer_D_LR(d_lr_grads)

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * dataset.get_dataset_size() + i
        batches_left = opt.n_epochs * dataset.get_dataset_size() - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
            f'\r[Epoch {epoch}/{opt.n_epochs}] '
            f'[Batch {i}/{dataset.get_dataset_size()}] '
            f'[D VAE loss: {loss_D_VAE.asnumpy().item():.4f}, '
            f'LR loss:{loss_D_LR.asnumpy().item():.4f}] '
            f'[G loss: {loss_GE.asnumpy().item():.4f}, '
            f'pixel: {loss_pixel.asnumpy().item():.4f}, '
            f'kl: {loss_kl.asnumpy().item():.4f}, '
            f'latent:{loss_latent.asnumpy().item():.4f}] '
            f'ETA: {time_left}'
        )

        if batches_done % opt.sample_interval == 0:
            sample_image(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            mindspore.save_checkpoint(generator, f'saved_models/{opt.dataset_name}/generator_{epoch}.ckpt')
            mindspore.save_checkpoint(encoder, f'saved_models/{opt.dataset_name}/encoder_{epoch}.ckpt')
            mindspore.save_checkpoint(D_VAE, f'saved_models/{opt.dataset_name}/D_VAE_{epoch}.ckpt')
            mindspore.save_checkpoint(D_LR, f'saved_models/{opt.dataset_name}/D_LR_{epoch}.ckpt')
