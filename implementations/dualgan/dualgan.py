"""DualGAN Model"""

import argparse
import datetime
import itertools
import os
import sys
import time

import mindspore
import numpy as np
from mindspore import nn, Tensor
from mindspore import ops
from mindspore.dataset.vision import transforms, Inter
import mindspore.common.dtype as mstype

from datasets import Edges2ShoesDataset
from img_utils import to_image
from models import Generator, Discriminator

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs(f'images/{opt.dataset_name}', exist_ok=True)
os.makedirs(f'saved_models/{opt.dataset_name}', exist_ok=True)

# Losses
cycle_loss = nn.L1Loss()

# Loss weights
lambda_adv = 1
lambda_cycle = 10
lambda_gp = 10

# Initialize generator and discriminator
G_AB = Generator()
G_BA = Generator()
D_A = Discriminator()
D_B = Discriminator()

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
    transforms.Resize((opt.img_size, opt.img_size), Inter.BICUBIC),
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


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random(size=(real_samples.shape[0], 1, 1, 1)),dtype=mstype.float32)
    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    # Get gradient w.r.t. interpolates
    grad_fn = mindspore.grad(D, return_ids=True)
    gradients = grad_fn(interpolates)
    gradients = mindspore.get_grad(gradients, 0)
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sample_image(batches):
    """Saves a generated sample from the validation set"""
    imgs = next(val_dataset.create_tuple_iterator())
    G_AB.set_train(False)
    G_BA.set_train(False)
    _real_A = imgs[0]
    _fake_B = G_AB(_real_A)
    AB = ops.cat((_real_A, _fake_B), -2)
    _real_B = imgs[1]
    _fake_A = G_BA(_real_B)
    BA = ops.cat((_real_B, _fake_A), -2)
    img_sample = ops.cat((AB, BA), 0)
    to_image(img_sample, os.path.join(f'images/{opt.dataset_name}', f'{batches}.png'))


def g_forward(_real_A, _real_B):
    """GeneratorUNet warmup forward func"""
    # Translate images to opposite domain
    _fake_A = G_BA(_real_B)
    _fake_B = G_AB(_real_A)

    # Reconstruct images
    recov_A = G_BA(_fake_B)
    recov_B = G_AB(_fake_A)

    # Adversarial loss
    _G_adv = -ops.mean(D_A(_fake_A)) - ops.mean(D_B(_fake_B))
    # Cycle loss
    _G_cycle = cycle_loss(recov_A, _real_A) + cycle_loss(recov_B, _real_B)
    # Total loss
    _G_loss = lambda_adv * _G_adv + lambda_cycle * _G_cycle

    return _G_loss, _G_cycle, _G_adv


def d_a_forward(_real_A, _fake_A):
    """Discriminator A forward function"""

    # Compute gradient penalty for improved wasserstein training
    gp_A = compute_gradient_penalty(D_A, _real_A, _fake_A)
    # Adversarial loss
    _D_A_loss = -ops.mean(D_A(_real_A)) + ops.mean(D_A(_fake_A)) + lambda_gp * gp_A

    return _D_A_loss


def d_b_forward(_real_B, _fake_B):
    """Discriminator B forward function"""

    # Compute gradient penalty for improved wasserstein training
    gp_B = compute_gradient_penalty(D_B, _real_B, _fake_B)
    # Adversarial loss
    _D_B_loss = -ops.mean(D_B(_real_B)) + ops.mean(D_B(_fake_B)) + lambda_gp * gp_B

    return _D_B_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d_a = ops.value_and_grad(d_a_forward, None, optimizer_D_A.parameters, has_aux=False)
grad_d_b = ops.value_and_grad(d_b_forward, None, optimizer_D_B.parameters, has_aux=False)

# ----------
#  Training
# ----------

batches_done = 0
prev_time = time.time()
for epoch in range(opt.n_epochs):
    # Model inputs
    for i, (real_A, real_B) in enumerate(dataset.create_tuple_iterator()):

        # ----------------------
        #  Train Discriminators
        # ----------------------

        D_A.set_train()
        D_B.set_train()

        # Generate a batch of images
        fake_A = ops.stop_gradient(G_BA(real_B))
        fake_B = ops.stop_gradient(G_AB(real_A))

        # ----------
        # Domain A
        # ----------

        (D_A_loss), d_a_grads = grad_d_a(real_A, fake_A)
        optimizer_D_A(d_a_grads)

        # ----------
        # Domain B
        # ----------

        (D_B_loss), d_b_grads = grad_d_b(real_B, fake_B)
        optimizer_D_B(d_b_grads)

        # Total loss
        D_loss = D_A_loss + D_B_loss

        if i % opt.n_critic == 0:
            # ------------------
            #  Train Generators
            # ------------------

            G_AB.set_train()
            G_BA.set_train()

            (G_loss, G_cycle, G_adv), g_grads = grad_g(real_A, real_B)
            optimizer_G(g_grads)

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_left = opt.n_epochs * dataset.get_dataset_size() - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / opt.n_critic)
            prev_time = time.time()

            sys.stdout.write(
                f'\r[Epoch {epoch}/{opt.n_epochs}] '
                f'[Batch {i}/{dataset.get_dataset_size()}] '
                f'[D loss: {D_loss.asnumpy().item():.4f}] '
                f'[G loss: {G_adv.asnumpy().item():.4f}, '
                f'cycle:{G_cycle.asnumpy().item():.4f}] '
                f'ETA: {time_left}'
            )

        if batches_done % opt.sample_interval == 0:
            sample_image(batches_done)

        batches_done += 1

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            mindspore.save_checkpoint(G_AB, f'saved_models/{opt.dataset_name}/G_AB_{epoch}.ckpt')
            mindspore.save_checkpoint(G_BA, f'saved_models/{opt.dataset_name}/G_BA_{epoch}.ckpt')
            mindspore.save_checkpoint(D_A, f'saved_models/{opt.dataset_name}/D_A_{epoch}.ckpt')
            mindspore.save_checkpoint(D_B, f'saved_models/{opt.dataset_name}/D_B_{epoch}.ckpt')
