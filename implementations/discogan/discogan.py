"""DiscoGAN Model"""

import argparse
import datetime
import itertools
import os
import sys
import time

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.dataset.vision import transforms, Inter

from datasets import Edges2ShoesDataset
from img_utils import to_image
from models import GeneratorUNet, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=64, help="size of image height")
parser.add_argument("--img_width", type=int, default=64, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs(f'images/{opt.dataset_name}', exist_ok=True)
os.makedirs(f'saved_models/{opt.dataset_name}', exist_ok=True)

# Losses
adversarial_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()
pixelwise_loss = nn.L1Loss()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorUNet(input_shape)
G_BA = GeneratorUNet(input_shape)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

G_AB.update_parameters_name("G_AB")
G_BA.update_parameters_name("G_BA")
D_A.update_parameters_name("G_AB")
D_B.update_parameters_name("G_BA")

if opt.epoch != 0:
    # Load pretrained models
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/G_AB_{opt.epoch}.ckpt', G_AB)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/G_BA_{opt.epoch}.ckpt', G_BA)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/D_A_{opt.epoch}.ckpt', D_A)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/D_B_{opt.epoch}.ckpt', D_B)

# Optimizers
optimizer_G = nn.optim.Adam(itertools.chain(G_AB.trainable_params(), G_BA.trainable_params()),
                            learning_rate=opt.lr,
                            beta1=opt.b1, beta2=opt.b2)
optimizer_D_A = nn.optim.Adam(D_A.trainable_params(), learning_rate=opt.lr,
                              beta1=opt.b1, beta2=opt.b2)

optimizer_D_B = nn.optim.Adam(D_B.trainable_params(), learning_rate=opt.lr,
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
).batch(16)


def sample_image(batches):
    """Saves a generated sample from the validation set"""
    imgs = next(val_dataset.create_tuple_iterator())
    G_AB.set_train(False)
    G_BA.set_train(False)
    _real_A = imgs[0]
    _fake_B = G_AB(_real_A)
    _real_B = imgs[1]
    _fake_A = G_BA(_real_B)
    img_sample = ops.cat((_real_A, _fake_B, _real_B, _fake_A), 0)
    to_image(img_sample, os.path.join(f'images/{opt.dataset_name}', f'{batches}.png'))


def g_forward(_real_A, _real_B, _valid):
    """Generator warmup forward func"""
    # GAN loss
    _fake_B = G_AB(_real_A)
    loss_GAN_AB = adversarial_loss(D_B(_fake_B), _valid)
    _fake_A = G_BA(_real_B)
    loss_GAN_BA = adversarial_loss(D_A(_fake_A), _valid)

    _loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

    # Pixelwise translation loss
    _loss_pixelwise = (pixelwise_loss(_fake_A, _real_A) +
                       pixelwise_loss(_fake_B, _real_B)) / 2

    # Cycle loss
    loss_cycle_A = cycle_loss(G_BA(_fake_B), _real_A)
    loss_cycle_B = cycle_loss(G_AB(_fake_A), _real_B)
    _loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

    # Total loss
    _loss_G = _loss_GAN + _loss_cycle + _loss_pixelwise

    return _loss_G, _loss_GAN, _loss_pixelwise, _loss_cycle, _fake_A, _fake_B


def d_a_forward(_real_A, _fake_A, _valid, _fake):
    """Discriminator A forward function"""

    # Real loss
    loss_real = adversarial_loss(D_A(_real_A), _valid)
    # Fake loss (on batch of previously generated samples)
    loss_fake = adversarial_loss(D_A(_fake_A), _fake)
    # Total loss
    _loss_D_A = (loss_real + loss_fake) / 2

    return _loss_D_A


def d_b_forward(_real_B, _fake_B, _valid, _fake):
    """Discriminator B forward function"""

    # Real loss
    loss_real = adversarial_loss(D_A(_real_B), _valid)
    # Fake loss (on batch of previously generated samples)
    loss_fake = adversarial_loss(D_A(_fake_B), _fake)
    # Total loss
    _loss_D_B = (loss_real + loss_fake) / 2

    return _loss_D_B


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d_a = ops.value_and_grad(d_a_forward, None, optimizer_D_A.parameters, has_aux=False)
grad_d_b = ops.value_and_grad(d_b_forward, None, optimizer_D_B.parameters, has_aux=False)

# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.n_epochs):
    # Model inputs
    for i, (real_A, real_B) in enumerate(dataset.create_tuple_iterator()):

        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((real_A.shape[0], *D_A.output_shape)))
        fake = ops.stop_gradient(ops.zeros((real_A.shape[0], *D_A.output_shape)))

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.set_train()
        G_BA.set_train()

        (loss_G, loss_GAN, loss_pixelwise, loss_cycle, fake_A, fake_B), g_grads = grad_g(real_A, real_B, valid)
        optimizer_G(g_grads)

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        D_A.set_train()

        (loss_D_A), d_a_grads = grad_d_a(real_A, ops.stop_gradient(fake_A), valid, fake)
        optimizer_D_A(d_a_grads)

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        D_B.set_train()

        (loss_D_B), d_b_grads = grad_d_b(real_B, ops.stop_gradient(fake_B), valid, fake)
        optimizer_D_B(d_b_grads)

        loss_D = 0.5 * (loss_D_A + loss_D_B)

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
            f'[D VAE loss: {loss_D.asnumpy().item():.4f}] '
            f'[G loss: {loss_G.asnumpy().item():.4f}, '
            f'adv: {loss_GAN.asnumpy().item():.4f}, '
            f'pixel: {loss_pixelwise.asnumpy().item():.4f}, '
            f'cycle:{loss_cycle.asnumpy().item():.4f}] '
            f'ETA: {time_left}'
        )

        if batches_done % opt.sample_interval == 0:
            sample_image(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            mindspore.save_checkpoint(G_AB, f'saved_models/{opt.dataset_name}/G_AB_{epoch}.ckpt')
            mindspore.save_checkpoint(G_BA, f'saved_models/{opt.dataset_name}/G_BA_{epoch}.ckpt')
            mindspore.save_checkpoint(D_A, f'saved_models/{opt.dataset_name}/D_A_{epoch}.ckpt')
            mindspore.save_checkpoint(D_B, f'saved_models/{opt.dataset_name}/D_B_{epoch}.ckpt')
