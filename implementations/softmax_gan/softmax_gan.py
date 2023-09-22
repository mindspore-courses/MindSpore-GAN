"""SoftmaxGAN Model"""

import argparse
import gzip
import os
import shutil
import urllib.request

import mindspore
import numpy as np
from mindspore import nn
from mindspore import ops
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
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


class Generator(nn.Cell):
    """Generator Network"""

    def __init__(self):
        super().__init__(Generator)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Dense(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8, affine=False))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.SequentialCell(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Dense(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def construct(self, _z):
        img = self.model(_z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self):
        super().__init__(Discriminator)

        self.model = nn.SequentialCell(
            nn.Dense(opt.img_size ** 2, 512),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dense(256, 1),
        )

    def construct(self, _img):
        img_flat = _img.view(_img.shape[0], -1)
        validity = self.model(img_flat)

        return validity


# Loss functions
adversarial_loss = nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

transform = [
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], is_hwc=False)
]

dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='train',
    shuffle=True
).map(operations=transform, input_columns="image").batch(opt.batch_size)

# Optimizers
optimizer_G = nn.optim.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)


def log(x):
    """Log function"""
    return ops.log(x + 1e-8)


def g_forward(_real_imgs, _g_target):
    """Generator forward function"""
    # Generate a batch of images
    z = ops.randn((_real_imgs.shape[0], opt.latent_dim), dtype=mstype.float32)
    _gen_imgs = generator(z)

    _d_real = discriminator(_real_imgs)
    _d_fake = discriminator(_gen_imgs)

    # Partition function
    Z = ops.sum(ops.exp(ops.neg(_d_real))) + ops.sum(ops.exp(ops.neg(_d_fake)))

    # Calculate loss of generator and update
    _g_loss = _g_target * (ops.sum(_d_real) + ops.sum(_d_fake)) + log(Z)

    return _g_loss, _gen_imgs


def d_forward(_real_imgs, _gen_imgs, _d_target):
    """Discriminator forward function"""

    _d_real = discriminator(_real_imgs)
    _d_fake = discriminator(_gen_imgs)

    # Partition function
    Z = ops.sum(ops.exp(ops.neg(_d_real))) + ops.sum(ops.exp(ops.neg(_d_fake)))

    # Calculate loss of discriminator and update
    _d_loss = _d_target * ops.sum(_d_real) + log(Z)

    return _d_loss, _d_real, _d_fake


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=True)

# ----------
#  Training
# ----------

generator.set_train()
discriminator.set_train()

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        g_target = 1 / (batch_size * 2)
        d_target = 1 / batch_size

        # Configure input
        real_imgs = imgs

        # -----------------
        #  Train Generator
        # -----------------

        (g_loss, gen_imgs), g_grads = grad_g(real_imgs, g_target)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        (d_loss, d_real, d_fake), d_grads = grad_d(real_imgs,
                                                   ops.stop_gradient(gen_imgs), d_target)
        optimizer_D(d_grads)

        # --------------
        # Log Progress
        # --------------

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] '
            f'[Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] '
            f'[G loss: {g_loss.asnumpy().item():.4f}]'
        )
        batches_done = epoch * dataset.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            to_image(gen_imgs[:25], os.path.join("images", F'{batches_done}.png'))
