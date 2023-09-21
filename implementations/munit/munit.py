"""MUNIT Model"""

import argparse
import datetime
import itertools
import os
import sys
import time

import mindspore
import mindspore.common.dtype as mstype
import numpy as np
from mindspore import nn, Tensor, ops
from mindspore.dataset.vision import transforms, Inter

from datasets import Edges2ShoesDataset
from img_utils import to_image
from models import Encoder, Decoder, MultiDiscriminator, LambdaLR

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=8, help="dimensionality of the style code")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs(f'images/{opt.dataset_name}', exist_ok=True)
os.makedirs(f'saved_models/{opt.dataset_name}', exist_ok=True)

criterion_recon = nn.L1Loss()

# Initialize encoders, generators and discriminators
Enc1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec1 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Enc2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec2 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
D1 = MultiDiscriminator()
D2 = MultiDiscriminator()

Enc1.update_parameters_name("Enc1")
Dec1.update_parameters_name("Dec1")
Enc2.update_parameters_name("Enc2")
Dec2.update_parameters_name("Dec2")
D1.update_parameters_name("D1")
D2.update_parameters_name("D2")

if opt.epoch != 0:
    # Load pretrained models
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/Enc1_{opt.epoch}.ckpt', Enc1)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/Dec1_{opt.epoch}.ckpt', Dec1)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/Enc2_{opt.epoch}.ckpt', Enc2)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/Dec2_{opt.epoch}.ckpt', Dec2)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/D1_{opt.epoch}.ckpt', D1)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/D2_{opt.epoch}.ckpt', D2)

# Loss weights
lambda_gan = 1
lambda_id = 10
lambda_style = 1
lambda_cont = 1
lambda_cyc = 0

# Learning rate update schedulers
decay_lr_G = LambdaLR(opt.lr, opt.n_epochs, opt.epoch, opt.decay_epoch, 49825)
decay_lr_D1 = LambdaLR(opt.lr, opt.n_epochs, opt.epoch, opt.decay_epoch, 49825)
decay_lr_D2 = LambdaLR(opt.lr, opt.n_epochs, opt.epoch, opt.decay_epoch, 49825)

# Optimizers
optimizer_G = nn.optim.Adam(itertools.chain(Enc1.trainable_params(), Dec1.trainable_params(),
                                            Enc2.trainable_params(), Dec2.trainable_params()),
                            learning_rate=decay_lr_G,
                            beta1=opt.b1, beta2=opt.b2)
optimizer_D1 = nn.optim.Adam(D1.trainable_params(), learning_rate=decay_lr_D1,
                             beta1=opt.b1, beta2=opt.b2)

optimizer_D2 = nn.optim.Adam(D2.trainable_params(), learning_rate=decay_lr_D2,
                             beta1=opt.b1, beta2=opt.b2)

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Inter.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False),
]

dataset = mindspore.dataset.GeneratorDataset(
    source=Edges2ShoesDataset(
        root="../../data/edges2shoes",
        transforms_=transforms_
    ),
    column_names=["A", "B"],
    shuffle=True
).batch(opt.batch_size)

val_dataset = mindspore.dataset.GeneratorDataset(
    source=Edges2ShoesDataset(
        root="../../data/edges2shoes",
        transforms_=transforms_,
        mode='val'
    ),
    column_names=["A", "B"],
    shuffle=True
).batch(5)


def sample_image(batches):
    """Saves a generated sample from the validation set"""
    imgs = next(val_dataset.create_tuple_iterator())
    img_samples = None
    Enc1.set_train(False)
    Dec2.set_train(False)
    for img1, _ in zip(imgs[0], imgs[1]):
        # Create copies of image
        _X1 = img1.unsqueeze(0).tile((opt.style_dim, 1, 1, 1))
        # Get random style codes
        s_code = np.random.uniform(-1, 1, (opt.style_dim, opt.style_dim))
        s_code = Tensor(s_code, dtype=mstype.float32)
        # Generate samples
        c_code_1, _ = Enc1(_X1)
        _X12 = Dec2(c_code_1, s_code)
        # Concatenate samples horisontally
        _X12 = ops.cat(list(X12), -1)
        img_sample = ops.cat((img1, _X12), -1).unsqueeze(0)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else ops.cat((img_samples, img_sample), -2)
    to_image(img_samples, os.path.join(f'images/{opt.dataset_name}', f'{batches}.png'))


def g_forward(_X1, _X2, _style_1, _style_2):
    """GeneratorUNet warmup forward func"""
    # Get shared latent representation
    c_code_1, s_code_1 = Enc1(_X1)
    c_code_2, s_code_2 = Enc2(_X2)

    # Reconstruct images
    _X11 = Dec1(c_code_1, s_code_1)
    _X22 = Dec2(c_code_2, s_code_2)

    # Translate images
    _X21 = Dec1(c_code_2, _style_1)
    _X12 = Dec2(c_code_1, _style_2)

    # Cycle translation
    c_code_21, s_code_21 = Enc1(_X21)
    c_code_12, s_code_12 = Enc2(_X12)
    X121 = Dec1(c_code_12, s_code_1) if lambda_cyc > 0 else 0
    X212 = Dec2(c_code_21, s_code_2) if lambda_cyc > 0 else 0

    # Losses
    loss_GAN_1 = lambda_gan * D1.compute_loss(_X21, valid)
    loss_GAN_2 = lambda_gan * D2.compute_loss(_X12, valid)
    loss_ID_1 = lambda_id * criterion_recon(_X11, _X1)
    loss_ID_2 = lambda_id * criterion_recon(_X22, _X2)
    loss_s_1 = lambda_style * criterion_recon(s_code_21, _style_1)
    loss_s_2 = lambda_style * criterion_recon(s_code_12, _style_2)
    loss_c_1 = lambda_cont * criterion_recon(c_code_12, ops.stop_gradient(c_code_1))
    loss_c_2 = lambda_cont * criterion_recon(c_code_21, ops.stop_gradient(c_code_2))
    loss_cyc_1 = lambda_cyc * criterion_recon(X121, _X1) if lambda_cyc > 0 else 0
    loss_cyc_2 = lambda_cyc * criterion_recon(X212, _X2) if lambda_cyc > 0 else 0

    # Total loss
    _loss_G = (
            loss_GAN_1
            + loss_GAN_2
            + loss_ID_1
            + loss_ID_2
            + loss_s_1
            + loss_s_2
            + loss_c_1
            + loss_c_2
            + loss_cyc_1
            + loss_cyc_2
    )

    return _loss_G, _X12, _X21


def d1_forward(_X1, _X21, _valid, _fake):
    """Discriminator A forward function"""

    _loss_D1 = D1.compute_loss(_X1, _valid) + D1.compute_loss(_X21, _fake)

    return _loss_D1


def d2_forward(_X2, _X12, _valid, _fake):
    """Discriminator B forward function"""

    _loss_D2 = D2.compute_loss(_X2, _valid) + D2.compute_loss(_X12, _fake)

    return _loss_D2


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d1 = ops.value_and_grad(d1_forward, None, optimizer_D1.parameters, has_aux=False)
grad_d2 = ops.value_and_grad(d2_forward, None, optimizer_D2.parameters, has_aux=False)

# ----------
#  Training
# ----------

# Adversarial ground truths
valid = 1
fake = 0

prev_time = time.time()
for epoch in range(opt.n_epochs):
    # Model inputs
    for i, (X1, X2) in enumerate(dataset.create_tuple_iterator()):
        style_1 = ops.randn((X1.shape[0], opt.style_dim, 1, 1), dtype=mstype.float32)
        style_2 = ops.randn((X1.shape[0], opt.style_dim, 1, 1), dtype=mstype.float32)

        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        Enc1.set_train()
        Enc2.set_train()
        Dec1.set_train()
        Dec2.set_train()

        (loss_G, X12, X21), g_grads = grad_g(X1, X2, style_1, style_2)
        optimizer_G(g_grads)

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        D1.set_train()

        (loss_D1), d1_grads = grad_d1(X1, ops.stop_gradient(X21), valid, fake)
        optimizer_D1(d1_grads)

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        D2.set_train()

        (loss_D2), d2_grads = grad_d2(X2, ops.stop_gradient(X12), valid, fake)
        optimizer_D2(d2_grads)

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
            f'[D loss: {(loss_D1.asnumpy().item() + loss_D2.asnumpy().item()):.4f}] '
            f'[G loss: {loss_G.asnumpy().item():.4f}] '
            f'ETA: {time_left}'
        )

        if batches_done % opt.sample_interval == 0:
            sample_image(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        mindspore.save_checkpoint(Enc1, f'saved_models/{opt.dataset_name}/Enc1_{epoch}.ckpt')
        mindspore.save_checkpoint(Dec1, f'saved_models/{opt.dataset_name}/Dec1_{epoch}.ckpt')
        mindspore.save_checkpoint(Enc2, f'saved_models/{opt.dataset_name}/Enc2_{epoch}.ckpt')
        mindspore.save_checkpoint(Dec2, f'saved_models/{opt.dataset_name}/Dec2_{epoch}.ckpt')
        mindspore.save_checkpoint(D1, f'saved_models/{opt.dataset_name}/D1_{epoch}.ckpt')
        mindspore.save_checkpoint(D2, f'saved_models/{opt.dataset_name}/D2_{epoch}.ckpt')
