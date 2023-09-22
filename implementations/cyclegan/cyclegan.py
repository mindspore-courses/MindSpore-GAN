"""CycleGAN Model"""

import argparse
import datetime
import os
import sys
import time
import urllib.request
import zipfile

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.dataset import MindDataset

from img_utils import to_image, make_grid
from models import GeneratorResNet, Discriminator
from utils import DynamicDecayLR, ReplayBuffer

file_path = "../../data/"

if not os.path.exists(file_path + "apple2orange"):
    # 下载数据集
    if not os.path.exists('../../data'):
        os.mkdir('../../data')
    os.mkdir(file_path + "apple2orange")
    url = 'https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/application/CycleGAN_apple2orange.zip'
    print("Downloading apple2orange dataset from " + url)
    urllib.request.urlretrieve(url, os.path.join(file_path, 'apple2orange.zip'))

    while not os.path.exists(os.path.join(file_path, 'apple2orange.zip')):
        pass

    zippedData = zipfile.ZipFile(os.path.join(file_path, 'apple2orange.zip'), 'r')
    print("Unzipping dataset...")
    zippedData.extractall(path=file_path)
    zippedData.close()
    os.rename(file_path + 'CycleGAN_apple2orange', file_path + 'apple2orange')
    os.remove(os.path.join(file_path, 'apple2orange.zip'))

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="apple2orange", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs(f'images/{opt.dataset_name}', exist_ok=True)
os.makedirs(f'saved_models/{opt.dataset_name}', exist_ok=True)

# Losses
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
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

# Learning rate update schedulers
decay_lr_G_AB = DynamicDecayLR(opt.lr, opt.n_epochs, 1019, opt.epoch, opt.decay_epoch)
decay_lr_G_BA = DynamicDecayLR(opt.lr, opt.n_epochs, 1019, opt.epoch, opt.decay_epoch)
decay_lr_D_A = DynamicDecayLR(opt.lr, opt.n_epochs, 1019, opt.epoch, opt.decay_epoch)
decay_lr_D_B = DynamicDecayLR(opt.lr, opt.n_epochs, 1019, opt.epoch, opt.decay_epoch)

optimizer_G_AB = nn.optim.Adam(G_AB.trainable_params(), learning_rate=decay_lr_G_AB,
                               beta1=opt.b1, beta2=opt.b2)
optimizer_G_BA = nn.optim.Adam(G_BA.trainable_params(), learning_rate=decay_lr_G_BA,
                               beta1=opt.b1, beta2=opt.b2)
optimizer_D_A = nn.optim.Adam(D_A.trainable_params(), learning_rate=decay_lr_D_A,
                              beta1=opt.b1, beta2=opt.b2)

optimizer_D_B = nn.optim.Adam(D_B.trainable_params(), learning_rate=decay_lr_D_B,
                              beta1=opt.b1, beta2=opt.b2)

optimizer_G_AB.update_parameters_name("optimizer_G_AB")
optimizer_G_BA.update_parameters_name("optimizer_G_BA")
optimizer_D_A.update_parameters_name("optimizer_D_A")
optimizer_D_B.update_parameters_name("optimizer_D_B")

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

dataset = MindDataset("../../data/apple2orange/apple2orange_train.mindrecord", shuffle=True).batch(opt.batch_size)

val_dataset = MindDataset("../../data/apple2orange/apple2orange_train.mindrecord", shuffle=True).batch(5)


def sample_images(batches):
    """Saves a generated sample from the validation set"""
    imgs = next(val_dataset.create_dict_iterator())
    G_AB.set_train(False)
    G_BA.set_train(False)
    _real_A = imgs["image_A"]
    _fake_B = G_AB(_real_A)
    _real_B = imgs["image_B"]
    _fake_A = G_BA(_real_B)
    # Arange images along x-axis
    _real_A = make_grid(_real_A, nrow=5, normalize=True)
    _real_B = make_grid(_real_B, nrow=5, normalize=True)
    _fake_A = make_grid(_fake_A, nrow=5, normalize=True)
    _fake_B = make_grid(_fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = ops.cat((_real_A, _fake_B, _real_B, _fake_A), 1)
    to_image(image_grid, os.path.join(f'images/{opt.dataset_name}', F'{batches}.png'))


def g_forward(_real_A, _real_B, _valid):
    """Generator forward function"""
    _fake_A = G_BA(_real_B)
    _fake_B = G_AB(_real_A)
    _rec_A = G_BA(_fake_B)
    _rec_B = G_AB(_fake_A)
    _identity_A = G_BA(_real_A)
    _identity_B = G_AB(_real_B)
    _loss_G_A = criterion_GAN(D_B(_fake_B), _valid)
    _loss_G_B = criterion_GAN(D_A(_fake_A), _valid)
    _loss_cycle_A = criterion_cycle(_rec_A, _real_A) * opt.lambda_cyc
    _loss_cycle_B = criterion_cycle(_rec_B, _real_B) * opt.lambda_cyc
    _loss_idt_A = criterion_identity(_identity_A, _real_A) * opt.lambda_cyc * opt.lambda_id
    _loss_idt_B = criterion_identity(_identity_B, _real_B) * opt.lambda_cyc * opt.lambda_id
    loss_G_ = _loss_G_A + _loss_G_B + _loss_cycle_A + _loss_cycle_B + _loss_idt_A + _loss_idt_B

    return loss_G_, _loss_G_A, _loss_G_B, _loss_cycle_A, _loss_cycle_B, _loss_idt_A, _loss_idt_B, _fake_A, _fake_B


def d_a_forward(_real_A, _fake_A, _valid, _fake):
    """Discriminator forward function"""
    # Real loss
    loss_real_ = criterion_GAN(D_A(_real_A), _valid)
    # Fake loss (on batch of previously generated samples)
    _fake_A_ = fake_A_buffer.push_and_pop(_fake_A)
    loss_fake_ = criterion_GAN(D_A(ops.stop_gradient(_fake_A_)), _fake)
    # Total loss
    loss_D_A_ = (loss_real_ + loss_fake_) / 2

    return loss_D_A_


def d_b_forward(_real_B, _fake_B, _valid, _fake):
    """Discriminator forward function"""
    # Real loss
    loss_real_ = criterion_GAN(D_B(_real_B), _valid)
    # Fake loss (on batch of previously generated samples)
    _fake_B_ = fake_B_buffer.push_and_pop(_fake_B)
    loss_fake_ = criterion_GAN(D_B(ops.stop_gradient(_fake_B_)), _fake)
    # Total loss
    loss_D_B_ = (loss_real_ + loss_fake_) / 2

    return loss_D_B_


grad_g_ab = ops.value_and_grad(g_forward, None, optimizer_G_AB.parameters)
grad_g_ba = ops.value_and_grad(g_forward, None, optimizer_G_BA.parameters)
grad_d_a = ops.value_and_grad(d_a_forward, None, optimizer_D_A.parameters)
grad_d_b = ops.value_and_grad(d_b_forward, None, optimizer_D_B.parameters)

# ----------
#  Training
# ----------

G_AB.set_train()
G_BA.set_train()
D_A.set_train()
D_B.set_train()

prev_time = time.time()
for epoch in range(opt.n_epochs):
    # Model inputs
    for i, batch in enumerate(dataset.create_dict_iterator()):
        # Set model input
        real_A = batch["image_A"]
        real_B = batch["image_B"]

        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((real_A.shape[0], *D_A.output_shape)))
        fake = ops.stop_gradient(ops.zeros((real_A.shape[0], *D_A.output_shape)))

        # ------------------
        #  Train Generators
        # ------------------

        loss_G, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_idt_B, fake_A, fake_B = g_forward(
            real_A, real_B, valid)

        (_, _, _, _, _, _, _, _, _), g_ab_grads = grad_g_ab(real_A, real_B, valid)
        (_, _, _, _, _, _, _, _, _), g_ba_grads = grad_g_ba(real_A, real_B, valid)
        optimizer_G_AB(g_ab_grads)
        optimizer_G_BA(g_ba_grads)

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        (loss_D_A), d_a_grads = grad_d_a(real_A, fake_A, valid, fake)
        optimizer_D_A(d_a_grads)

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        (loss_D_B), d_b_grads = grad_d_b(real_B, fake_B, valid, fake)
        optimizer_D_B(d_b_grads)

        loss_D = (loss_D_A + loss_D_B) / 2

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
            f'[D loss: {loss_D.asnumpy().item():.4f}] '
            f'[G loss: {loss_G.asnumpy().item():.4f}, '
            f'adv: A: {loss_G_A.asnumpy().item():.4f} B: {loss_G_B.asnumpy().item():.4f}, '
            f'cycle: A: {loss_cycle_A.asnumpy().item():.4f}, B: {loss_cycle_B.asnumpy().item():.4f}, '
            f'identity: A: {loss_idt_A.asnumpy().item():.4f}, B: {loss_idt_B.asnumpy().item():4f}] '
            f'ETA: {time_left}'
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        mindspore.save_checkpoint(G_AB, f'saved_models/{opt.dataset_name}/G_AB_{epoch}.ckpt')
        mindspore.save_checkpoint(G_BA, f'saved_models/{opt.dataset_name}/G_BA_{epoch}.ckpt')
        mindspore.save_checkpoint(D_A, f'saved_models/{opt.dataset_name}/D_A_{epoch}.ckpt')
        mindspore.save_checkpoint(D_B, f'saved_models/{opt.dataset_name}/D_B_{epoch}.ckpt')
