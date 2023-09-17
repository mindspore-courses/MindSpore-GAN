"""
StarGAN (CelebA)
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
And the annotations: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt
Instructions on running the script:
1. Download the dataset and annotations from the provided link
2. Copy 'list_attr_celeba.txt' to folder 'img_align_celeba'
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the script by 'python3 stargan.py'
"""

import argparse
import datetime
import os
import sys
import time

import mindspore
import mindspore.common.dtype as mstype
from mindspore import nn, Tensor
from mindspore import ops
from mindspore.dataset import CelebADataset
from mindspore.dataset.vision import transforms, Inter

from img_utils import to_image
from models import GeneratorResNet, Discriminator

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument(
    "--selected_attrs",
    "--list",
    nargs="+",
    help="selected attributes for the CelebA dataset",
    default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
)
parser.add_argument("--n_critic", type=int, default=5, help="number of training iterations for WGAN discriminator")
opt = parser.parse_args()
print(opt)

mindspore.set_context(mode=0)

c_dim = 40
img_shape = (opt.channels, opt.img_height, opt.img_width)

criterion_cycle = nn.L1Loss()


def criterion_cls(logit, target):
    """cls criterion"""
    return (ops.binary_cross_entropy_with_logits(logit, target, weight=ops.ones_like(logit), reduction='mean',
                                                 pos_weight=ops.ones_like(logit)) / logit.shape[0])


# Loss weights
lambda_cls = 1
lambda_rec = 10
lambda_gp = 10

# Initialize generator and discriminator
generator = GeneratorResNet(img_shape=img_shape, res_blocks=opt.residual_blocks, c_dim=c_dim)

discriminator = Discriminator(img_shape=img_shape, c_dim=c_dim)

if opt.epoch != 0:
    # Load pretrained models
    mindspore.load_checkpoint(f'saved_models/generator_{opt.epoch}.ckpt', generator)
    mindspore.load_checkpoint(f'saved_models/discriminator{opt.epoch}.ckpt', discriminator)

# Optimizers
optimizer_G = nn.optim.Adam(generator.trainable_params(), opt.lr, opt.b1, opt.b2)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), opt.lr, opt.b1, opt.b2)

train_transforms = [
    transforms.Resize(int(1.12 * opt.img_height), Inter.BICUBIC),
    transforms.RandomCrop(opt.img_height),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False),
]

val_transforms = [
    transforms.Resize((opt.img_height, opt.img_width), Inter.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False),
]

train_dataset = (CelebADataset(
    dataset_dir="../../data/CelebA",
    shuffle=True,
    decode=True,
).map(operations=train_transforms, input_columns=["image"])
                 .batch(opt.batch_size))

val_dataset = CelebADataset(
    dataset_dir="../../data/CelebA",
    shuffle=True,
    decode=True
).map(operations=val_transforms, input_columns=["image"]).batch(10)


def compute_gradient_penalty(real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = ops.rand((real_samples.shape[0], 1, 1, 1))
    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    # Get gradient w.r.t. interpolates
    grad_fn = ops.grad(discriminator)
    gradients = grad_fn(interpolates)
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


label_changes = [
    ((0, 1), (1, 0), (2, 0)),  # Set to black hair
    ((0, 0), (1, 1), (2, 0)),  # Set to blonde hair
    ((0, 0), (1, 0), (2, 1)),  # Set to brown hair
    ((3, -1),),  # Flip gender
    ((4, -1),),  # Age flip
]


def sample_images(_batches_done):
    """Saves a generated sample of domain translations"""
    val_imgs, val_labels = next(val_dataset.create_tuple_iterator())

    img_samples = None
    for j in range(10):
        img, label = val_imgs[j], val_labels[j]
        # Repeat for number of label changes
        _imgs = img.tile((c_dim, 1, 1, 1))
        _labels = label.tile((c_dim, 1))
        # Make changes to labels
        for sample_i, changes in enumerate(label_changes):
            for col, val in changes:
                if val == -1:
                    _labels[sample_i, col] = 1 - _labels[sample_i, col].asnumpy().item()
                else:
                    _labels[sample_i, col] = val

        # Generate translations
        _labels = ops.Cast()(_labels, mstype.float32)
        gen_imgs = generator(_imgs, _labels)
        # Concatenate images by width
        gen_imgs = ops.cat(list(gen_imgs), -1)
        img_sample = ops.cat((img, gen_imgs), -1)
        # Add as row to generated samples
        img_samples = img_sample if img_samples is None else ops.cat((img_samples, img_sample), -2)

    to_image(img_samples.view(1, *img_samples.shape), f'images/{_batches_done}.png')


def g_forward(_imgs, _labels, _sampled_c):
    """Generator warmup forward func"""
    # Translate and reconstruct image
    gen_imgs = generator(_imgs, _sampled_c)
    recov_imgs = generator(gen_imgs, _labels)
    # Discriminator evaluates translated image
    fake_validity, pred_cls = discriminator(gen_imgs)
    # Adversarial loss
    _g_loss_adv = -ops.mean(fake_validity)
    # Classification loss
    _g_loss_cls = criterion_cls(pred_cls, _sampled_c)
    # Reconstruction loss
    _g_loss_rec = criterion_cycle(recov_imgs, _imgs)
    # Total loss
    _g_loss = _g_loss_adv + lambda_cls * _g_loss_cls + lambda_rec * _g_loss_rec

    return _g_loss, _g_loss_adv, _g_loss_cls, _g_loss_rec


def d_forward(_imgs, _fake_imgs, _labels):
    """Discriminator forward function"""
    # Real images
    real_validity, pred_cls = discriminator(_imgs)
    pred_cls = ops.Cast()(pred_cls, mstype.float32)
    # Fake images
    fake_validity, _ = discriminator(_fake_imgs)
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(_imgs, _fake_imgs)
    # Adversarial loss
    _d_loss_adv = -ops.mean(real_validity) + ops.mean(fake_validity) + lambda_gp * gradient_penalty
    # Classification loss
    _d_loss_cls = criterion_cls(pred_cls, _labels)
    # Total loss
    _d_loss = _d_loss_adv + lambda_cls * _d_loss_cls
    return _d_loss, _d_loss_adv, _d_loss_cls


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=True)

# ----------
#  Training
# ----------

generator.set_train()
discriminator.set_train()

saved_samples = []
start_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    # Model inputs
    for i, (imgs, labels) in enumerate(train_dataset.create_tuple_iterator()):
        # Sample labels as generator inputs
        sampled_c = ops.randint(0, 2, (imgs.shape[0], c_dim))
        sampled_c = ops.Cast()(sampled_c, mstype.float32)
        labels = ops.Cast()(labels, mstype.float32)
        # Generate fake batch of images
        fake_imgs = generator(imgs, sampled_c)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        (d_loss, d_loss_adv, d_loss_cls), d_grads = grad_d(imgs, ops.stop_gradient(fake_imgs), labels)
        optimizer_D(d_grads)

        # Every n_critic times update generator
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            (g_loss, g_loss_adv, g_loss_cls, g_loss_rec), g_grads = grad_g(imgs, labels, sampled_c)
            optimizer_G(g_grads)

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * train_dataset.get_dataset_size() + i
            batches_left = opt.n_epochs * train_dataset.get_dataset_size() - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time) / (batches_done + 1))

            # Print log
            sys.stdout.write(
                f'\r[Epoch {epoch}/{opt.n_epochs}] '
                f'[Batch {i}/{train_dataset.get_dataset_size()}] '
                f'[D adv: {d_loss_adv.asnumpy().item():.4f}, '
                f'aux: {d_loss_cls.asnumpy().item():.4f}] '
                f'[G loss: {g_loss.asnumpy().item():.4f}, '
                f'adv: {g_loss_adv.asnumpy().item():.4f}, '
                f'aux: {g_loss_cls.asnumpy().item():.4f}, '
                f'cycle: {g_loss_rec.asnumpy().item():.4f}, '
                f'ETA: {time_left}"'
            )

            # If at sample interval sample and save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            mindspore.save_checkpoint(generator, f'saved_models/generator_{epoch}.ckpt')
            mindspore.save_checkpoint(discriminator, f'saved_models/discriminator_{epoch}.ckpt')
