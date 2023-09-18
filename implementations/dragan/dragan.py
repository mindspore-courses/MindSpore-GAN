"""DRAGAN Model"""

import argparse
import gzip
import os
import shutil
import urllib.request

import mindspore
import mindspore.common.initializer as init
import numpy as np
from mindspore import Tensor, ops
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import transforms

from img_utils import to_image

file_path = "../../data/MNIST/"

if not os.path.exists(file_path):
    # 下载数据集
    if not os.path.exists('../../data'):
        os.mkdir('../../data')
    os.mkdir(file_path)
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (base_url + file_name).format(**locals())
        print("Downloading MNIST dataset from" + url)
        urllib.request.urlretrieve(url, os.path.join(file_path, file_name))
        with gzip.open(os.path.join(file_path, file_name), 'rb') as f_in:
            print("Unzipping...")
            with open(os.path.join(file_path, file_name)[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(os.path.join(file_path, file_name))

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
opt = parser.parse_args()
print(opt)


class Generator(nn.Cell):
    """Generator Network"""

    def __init__(self):
        super().__init__(Generator)

        self.init_size = opt.img_size // 4
        self.l1 = nn.SequentialCell(
            nn.Dense(opt.latent_dim, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.SequentialCell(
            nn.BatchNorm2d(128,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, 128, 3, stride=1,
                      pad_mode='pad', padding=1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(128, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, 64, 3, stride=1,
                      pad_mode='pad', padding=1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(64, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, opt.channels, 3, stride=1,
                      pad_mode='pad', padding=1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh(),
        )

    def construct(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self):
        super().__init__(Discriminator)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2,
                          pad_mode='pad', padding=1, has_bias=False,
                          weight_init=init.Normal(0.02, 0.0)),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8,
                                            gamma_init=init.Normal(0.02, 1.0),
                                            beta_init=init.Constant(0.0), affine=False))
            return block

        self.model = nn.SequentialCell(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.SequentialCell(
            nn.Dense(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def construct(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = nn.BCELoss()

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

transform = [
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
]

dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='train',
    shuffle=True
).map(operations=transform, input_columns="image").batch(opt.batch_size)

# Optimizers
optimizer_G = nn.optim.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)


def compute_gradient_penalty(D, X):
    """Calculates the gradient penalty loss for DRAGAN"""
    # Random weight term for interpolation
    alpha = Tensor(np.random.random(size=X.shape))

    interpolates = alpha * X + ((1 - alpha) * (X + 0.5 * X.std() * ops.rand(X.shape)))
    interpolates = ops.Cast()(interpolates, mstype.float32)

    # Get gradient w.r.t. interpolates
    grad_fn = ops.grad(D)
    gradients = grad_fn(interpolates)[0]

    _gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return _gradient_penalty


def g_forward(_imgs, _valid):
    """Generator forward function"""
    # Sample noise as generator input
    z = ops.randn((_imgs.shape[0], opt.latent_dim), dtype=mstype.float32)

    # Generate a batch of images
    _gen_imgs = generator(z)

    # Loss measures generator's ability to fool the discriminator
    _g_loss = adversarial_loss(discriminator(_gen_imgs), _valid)
    return _g_loss, _gen_imgs


def d_forward(_real_imgs, _gen_imgs, _valid, _fake):
    """Discriminator forward function"""
    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(_real_imgs), _valid)
    fake_loss = adversarial_loss(discriminator(_gen_imgs), _fake)
    _d_loss = (real_loss + fake_loss) / 2

    # Calculate gradient penalty
    _gradient_penalty = compute_gradient_penalty(discriminator, _real_imgs)

    return _gradient_penalty, _d_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=True)

generator.set_train()
discriminator.set_train()

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        valid = ops.stop_gradient(ops.ones((imgs.shape[0], 1)))
        fake = ops.stop_gradient(ops.zeros((imgs.shape[0], 1)))

        # Configure input
        real_imgs = imgs

        # -----------------
        #  Train Generator
        # -----------------

        (g_loss, gen_imgs), g_grads = grad_g(real_imgs, valid)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        (gradient_penalty, d_loss), d_grads = grad_d(real_imgs, ops.stop_gradient(gen_imgs), valid, fake)
        optimizer_D(d_grads)

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] [G loss: {g_loss.asnumpy().item():.4f}]'
        )

    to_image(gen_imgs, os.path.join("images", F'{epoch}.png'))
