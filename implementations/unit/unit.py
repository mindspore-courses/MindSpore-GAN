"""UNIT Model"""

import argparse
import datetime
import itertools
import os
import sys
import time
import urllib
import zipfile

import mindspore
from mindspore import nn, ops
from mindspore.dataset import MindDataset
from mindspore.nn import CellList

from img_utils import to_image
from models import Encoder, ResidualBlock, Generator, LambdaLR, Discriminator

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
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs(f'images/{opt.dataset_name}', exist_ok=True)
os.makedirs(f'saved_models/{opt.dataset_name}', exist_ok=True)

# Losses
criterion_GAN = nn.MSELoss()
criterion_pixel = nn.L1Loss()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2 ** opt.n_downsample

# Initialize generator and discriminator
shared_E = ResidualBlock(features=shared_dim)
E1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
E2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
shared_G = ResidualBlock(features=shared_dim)
G1 = Generator(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
G2 = Generator(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
D1 = Discriminator(input_shape)
D2 = Discriminator(input_shape)

shared_E.update_parameters_name("shared_E_")
shared_G.update_parameters_name("shared_G_")
E1.update_parameters_name("E1_")
E2.update_parameters_name("E2_")
G1.update_parameters_name("G1_")
G2.update_parameters_name("G2_")
D1.update_parameters_name("D1_")
D2.update_parameters_name("D2_")

if opt.epoch != 0:
    # Load pretrained models
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/E1_{opt.epoch}.ckpt', E1)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/E2_{opt.epoch}.ckpt', E2)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/G1_{opt.epoch}.ckpt', G1)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/G2_{opt.epoch}.ckpt', G2)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/D1_{opt.epoch}.ckpt', D1)
    mindspore.load_checkpoint(f'saved_models/%{opt.dataset_name}/D2_{opt.epoch}.ckpt', D2)

# Loss weights
lambda_0 = 10  # GAN
lambda_1 = 0.1  # KL (encoded images)
lambda_2 = 100  # ID pixel-wise
lambda_3 = 0.1  # KL (encoded translated images)
lambda_4 = 100  # Cycle pixel-wise

# Learning rate update schedulers
decay_lr_G_AB = LambdaLR(opt.lr, opt.n_epochs, opt.epoch, opt.decay_epoch, 500)
decay_lr_G_BA = LambdaLR(opt.lr, opt.n_epochs, opt.epoch, opt.decay_epoch, 500)
decay_lr_D_A = LambdaLR(opt.lr, opt.n_epochs, opt.epoch, opt.decay_epoch, 500)
decay_lr_D_B = LambdaLR(opt.lr, opt.n_epochs, opt.epoch, opt.decay_epoch, 500)

no_shared_cell_E1 = CellList()
no_shared_cell_G1 = CellList()
for name, cell in E1.cells_and_names():
    if "shared_block" not in name and name != '':
        no_shared_cell_E1.append(cell)

for name, cell in G1.cells_and_names():
    if "shared_block" not in name and name != '':
        no_shared_cell_G1.append(cell)

no_shared_cell_E1.update_parameters_name("E1")
no_shared_cell_G1.update_parameters_name("G1")

optimizer_G = nn.optim.Adam(itertools.chain(no_shared_cell_E1.trainable_params(),
                                            E2.trainable_params(),
                                            no_shared_cell_G1.trainable_params(),
                                            G2.trainable_params()),
                            learning_rate=decay_lr_G_AB,
                            beta1=opt.b1, beta2=opt.b2)

optimizer_D1 = nn.optim.Adam(D1.trainable_params(), learning_rate=decay_lr_D_A,
                             beta1=opt.b1, beta2=opt.b2)

optimizer_D2 = nn.optim.Adam(D2.trainable_params(), learning_rate=decay_lr_D_B,
                             beta1=opt.b1, beta2=opt.b2)

dataset = (MindDataset("../../data/apple2orange/apple2orange_train.mindrecord", shuffle=True).
           batch(opt.batch_size))

val_dataset = (MindDataset("../../data/apple2orange/apple2orange_train.mindrecord", shuffle=True).
               batch(5))


def sample_images(batches):
    """Saves a generated sample from the test set"""
    imgs = next(val_dataset.create_dict_iterator())
    G1.set_train(False)
    G2.set_train(False)
    _X1 = imgs["image_A"]
    _X2 = imgs["image_B"]

    _, Z1 = E1(_X1)
    _, Z2 = E2(_X2)
    _fake_X1 = G1(Z2)
    _fake_X2 = G2(Z1)

    img_sample = ops.cat((_X1, _fake_X2, _X2, _fake_X1), 0)
    to_image(img_sample, os.path.join(f'images/{opt.dataset_name}', F'{batches}.png'))


def compute_kl(mu):
    """Compute KL"""
    mu_2 = ops.pow(mu, 2)
    loss = ops.mean(mu_2)
    return loss


def g_forward(_X1, _X2, _valid):
    """Generator forward function"""
    # Get shared latent representation
    mu1, Z1 = E1(_X1)
    mu2, Z2 = E2(_X2)

    # Reconstruct images
    recon_X1 = G1(Z1)
    recon_X2 = G2(Z2)

    # Translate images
    _fake_X1 = G1(Z2)
    _fake_X2 = G2(Z1)

    # Cycle translation
    mu1_, Z1_ = E1(_fake_X1)
    mu2_, Z2_ = E2(_fake_X2)
    cycle_X1 = G1(Z2_)
    cycle_X2 = G2(Z1_)

    # Losses
    loss_GAN_1 = lambda_0 * criterion_GAN(D1(_fake_X1), _valid)
    loss_GAN_2 = lambda_0 * criterion_GAN(D2(_fake_X2), _valid)
    loss_KL_1 = lambda_1 * compute_kl(mu1)
    loss_KL_2 = lambda_1 * compute_kl(mu2)
    loss_ID_1 = lambda_2 * criterion_pixel(recon_X1, _X1)
    loss_ID_2 = lambda_2 * criterion_pixel(recon_X2, _X2)
    loss_KL_1_ = lambda_3 * compute_kl(mu1_)
    loss_KL_2_ = lambda_3 * compute_kl(mu2_)
    loss_cyc_1 = lambda_4 * criterion_pixel(cycle_X1, _X1)
    loss_cyc_2 = lambda_4 * criterion_pixel(cycle_X2, _X2)

    # Total loss
    _loss_G = (
            loss_KL_1
            + loss_KL_2
            + loss_ID_1
            + loss_ID_2
            + loss_GAN_1
            + loss_GAN_2
            + loss_KL_1_
            + loss_KL_2_
            + loss_cyc_1
            + loss_cyc_2
    )

    return _loss_G, _fake_X1, _fake_X2


def d1_forward(_X1, _fake_X1, _valid, _fake):
    """Discriminator forward function"""

    _loss_D1 = criterion_GAN(D1(_X1), _valid) + criterion_GAN(D1(_fake_X1), _fake)

    return _loss_D1


def d2_forward(_X2, _fake_X2, _valid, _fake):
    """Discriminator forward function"""

    _loss_D2 = criterion_GAN(D2(X2), _valid) + criterion_GAN(D2(_fake_X2), _fake)

    return _loss_D2


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d1 = ops.value_and_grad(d1_forward, None, optimizer_D1.parameters)
grad_d2 = ops.value_and_grad(d2_forward, None, optimizer_D2.parameters)

# ----------
#  Training
# ----------

G1.set_train()
G2.set_train()
D1.set_train()
D2.set_train()
E1.set_train()
E2.set_train()

prev_time = time.time()
for epoch in range(opt.n_epochs):
    # Model inputs
    for i, batch in enumerate(dataset.create_dict_iterator()):
        # Set model input
        X1 = batch["image_A"]
        X2 = batch["image_B"]

        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((X1.shape[0], *D1.output_shape)))
        fake = ops.stop_gradient(ops.zeros((X1.shape[0], *D1.output_shape)))

        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        (loss_G, fake_X1, fake_X2), g_grads = grad_g(X1, X2, valid)
        optimizer_G(g_grads)

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        (loss_D1), d1_grads = grad_d1(X1, ops.stop_gradient(fake_X1), valid, fake)
        optimizer_D1(d1_grads)

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        (loss_D2), d2_grads = grad_d2(X2, ops.stop_gradient(fake_X2), valid, fake)
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

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        mindspore.save_checkpoint(E1, f'saved_models/{opt.dataset_name}/E1_{epoch}.ckpt')
        mindspore.save_checkpoint(E2, f'saved_models/{opt.dataset_name}/E2_{epoch}.ckpt')
        mindspore.save_checkpoint(G1, f'saved_models/{opt.dataset_name}/G1_{epoch}.ckpt')
        mindspore.save_checkpoint(G2, f'saved_models/{opt.dataset_name}/G2_{epoch}.ckpt')
        mindspore.save_checkpoint(D1, f'saved_models/{opt.dataset_name}/D1_{epoch}.ckpt')
        mindspore.save_checkpoint(D2, f'saved_models/{opt.dataset_name}/D2_{epoch}.ckpt')
