"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import os
import sys

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.dataset import CelebADataset
from mindspore.dataset.vision import transforms, Inter

from img_utils import to_image, make_grid
from models import GeneratorResNet, Discriminator, FeatureExtractor

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.set_train(False)

criterion_GAN = nn.MSELoss()
criterion_content = nn.L1Loss()

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

    # Adversarial loss
    loss_GAN = criterion_GAN(discriminator(_gen_hr), _valid)

    # Content loss
    gen_features = feature_extractor(_gen_hr)
    real_features = feature_extractor(_imgs_hr)
    loss_content = criterion_content(gen_features, ops.stop_gradient(real_features))

    # Total loss
    _g_loss = loss_content + 1e-3 * loss_GAN
    return _g_loss, _gen_hr


def d_forward(_imgs_hr, _gen_hr, _valid, _fake):
    """Discriminator forward function"""
    # Loss of real and fake images
    loss_real = criterion_GAN(discriminator(_imgs_hr), _valid)
    loss_fake = criterion_GAN(discriminator(_gen_hr), _fake)

    # Total loss
    _d_loss = (loss_real + loss_fake) / 2
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
        imgs_hr = batch["image_hr"]
        imgs_lr = batch["image_lr"]

        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((imgs_lr.shape[0], *discriminator.output_shape)))
        fake = ops.stop_gradient(ops.zeros((imgs_lr.shape[0], *discriminator.output_shape)))

        # ------------------
        #  Train Generators
        # ------------------
        (g_loss, gen_hr), g_grads = grad_g(imgs_hr, imgs_lr, valid)
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
            f'[G loss: {g_loss.asnumpy().item():.4f}]'
        )

        batches_done = epoch * dataset.get_dataset_size() + i
        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = ops.interpolate(imgs_lr, scale_factor=4.0, recompute_scale_factor=True)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = ops.cat((imgs_lr, gen_hr), -1)
            to_image(img_grid, f'images/{batches_done}.png')

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        mindspore.save_checkpoint(generator, f'saved_models/{opt.dataset_name}/generator_{epoch}.ckpt')
        mindspore.save_checkpoint(discriminator, f'saved_models/{opt.dataset_name}/discriminator_{epoch}.ckpt')
