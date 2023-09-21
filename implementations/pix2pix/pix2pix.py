"""Pix2Pix Model"""

import argparse
import datetime
import os
import sys
import tarfile
import time
import urllib.request

import mindspore
from mindspore import Tensor, ops
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.dataset import MindDataset
from mindspore.dataset.vision import transforms, Inter

from img_utils import to_image
from models import GeneratorUNet, Discriminator

file_path = "../../data/"

if not os.path.exists(file_path+"facades"):
    # 下载数据集
    if not os.path.exists('../../data'):
        os.mkdir('../../data')
    os.mkdir(file_path+"facades")
    url = 'https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/application/dataset_pix2pix.tar'
    print("Downloading facades dataset from " + url)
    urllib.request.urlretrieve(url, os.path.join(file_path, 'facades.tar'))
    tar = tarfile.open(os.path.join(file_path, 'facades.tar'), 'r')
    print("Unzipping dataset...")
    tar.extractall(path=file_path)
    tar.close()
    os.rename(file_path+'dataset_pix2pix',file_path+'facades')
    os.remove(os.path.join(file_path, 'facades.tar'))

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",
                    type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name",
                    type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size",
                    type=int, default=1, help="size of the batches")
parser.add_argument("--lr",
                    type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1",
                    type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2",
                    type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch",
                    type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu",
                    type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height",
                    type=int, default=256, help="size of image height")
parser.add_argument("--img_width",
                    type=int, default=256, help="size of image width")
parser.add_argument("--channels",
                    type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval",
                    type=int, default=10, help="interval between sampling of images from generators"
                    )
parser.add_argument("--checkpoint_interval",
                    type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs(f'images/{opt.dataset_name}', exist_ok=True)
os.makedirs(f'saved_models/{opt.dataset_name}', exist_ok=True)

# Loss functions
criterion_GAN = nn.MSELoss()
criterion_pixelwise = nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

optimizer_G = nn.optim.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)

transform = [
    transforms.Resize((opt.img_height, opt.img_width), Inter.BICUBIC),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.HWC2CHW()
]

dataset = MindDataset("../../data/facades/train.mindrecord",
                      columns_list=["input_images", "target_images"], shuffle=True)

val_dataset = MindDataset("../../data/facades/train.mindrecord",
                          columns_list=["input_images", "target_images"], shuffle=True)


def sample_image(batches):
    """Saves a generated sample from the validation set"""
    imgs = next(val_dataset.create_dict_iterator())
    _input_images = Tensor(imgs["input_images"], dtype=mstype.float32)
    _target_images = Tensor(imgs["target_images"], dtype=mstype.float32)
    fake_images = generator(_input_images)
    img_sample = ops.cat((_input_images, fake_images, _target_images), -2)
    to_image(img_sample, os.path.join(f'images/{opt.dataset_name}', F'{batches}.png'))


def g_forward(_real_A, _real_B, _valid):
    """GeneratorUNet forward function"""
    # GAN loss
    _fake_B = generator(_real_A)
    pred_fake = discriminator(_fake_B, _real_A)
    _loss_GAN = criterion_GAN(pred_fake, _valid)
    # Pixel-wise loss
    _loss_pixel = criterion_pixelwise(_fake_B, _real_B)
    # Total loss
    _g_loss = _loss_GAN + lambda_pixel * _loss_pixel
    return _g_loss, _fake_B, _loss_GAN, _loss_pixel


def d_forward(_real_A, _real_B, _fake_B, _valid, _fake):
    """Discriminator forward function"""
    # Real loss
    pred_real = discriminator(_real_B, _real_A)
    loss_real = criterion_GAN(pred_real, _valid)

    # Fake loss
    pred_fake = discriminator(_fake_B, _real_A)
    loss_fake = criterion_GAN(pred_fake, _fake)

    # Total loss
    _d_loss = 0.5 * (loss_real + loss_fake)
    return _d_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)

# ----------
#  Training
# ----------

prev_time = time.time()

generator.set_train()
discriminator.set_train()

for epoch in range(opt.n_epochs):
    # Model inputs
    for i, batch in enumerate(dataset.create_dict_iterator()):

        input_images = Tensor(batch["input_images"], dtype=mstype.float32)
        target_images = Tensor(batch["target_images"], dtype=mstype.float32)

        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((input_images.shape[0], *patch)))
        fake = ops.stop_gradient(ops.zeros((input_images.shape[0], *patch)))

        (g_loss, fake_imgs, GAN_loss, pixel_loss), g_grads = grad_g(input_images, target_images, valid)
        optimizer_G(g_grads)

        (d_loss), d_grads = grad_d(input_images, target_images, ops.stop_gradient(fake_imgs),
                                   valid, fake)
        optimizer_D(d_grads)

        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left
        batches_done = epoch * dataset.get_dataset_size() + i
        batches_left = opt.n_epochs * dataset.get_dataset_size() - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            f'\r[Epoch {epoch}/{opt.n_epochs}] '
            f'[Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] '
            f'[G loss: {g_loss.asnumpy().item():.4f}, '
            f'pixel: {pixel_loss.asnumpy().item():.4f}, adv: {GAN_loss.asnumpy().item():.4f}] '
            f'ETA: {time_left}"'
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_image(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        mindspore.save_checkpoint(generator, f'saved_models/{opt.dataset_name}/generator_{epoch}.ckpt')
        mindspore.save_checkpoint(discriminator, f'saved_models/{opt.dataset_name}/discriminator_{epoch}.ckpt')
