"""BGAN Model"""
# Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/


import argparse
import gzip
import os
import shutil
import urllib.request

import mindspore
import numpy as np
from mindspore import nn, Tensor
from mindspore import ops
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
        print("正在从" + url + "下载MNIST数据集...")
        urllib.request.urlretrieve(url, os.path.join(file_path, file_name))
        with gzip.open(os.path.join(file_path, file_name), 'rb') as f_in:
            print("正在解压数据集...")
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
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
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
        _img = self.model(_z)
        _img = _img.view(_img.shape[0], *img_shape)
        return _img


class Discriminator(nn.Cell):
    """Discriminator Network"""
    def __init__(self):
        super().__init__(Discriminator)

        self.model = nn.SequentialCell(
            nn.Dense(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dense(256, 1),
            nn.Sigmoid()
        )

    def construct(self, _img):
        img_flat = _img.view(_img.shape[0], -1)
        _validity = self.model(img_flat)
        return _validity


def boundary_seeking_loss(y_pred):
    """
    Boundary seeking loss.
    Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
    """
    return 0.5 * ops.mean((ops.log(y_pred) - ops.log(1 - y_pred)) ** 2)


discriminator_loss = nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

optimizer_G = nn.optim.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)

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


def g_forward(_z, _valid):
    """Generator forward function"""
    # Generate a batch of images
    _gen_imgs = generator(_z)
    # Loss measures generator's ability to fool the discriminator
    _g_loss = boundary_seeking_loss(discriminator(_gen_imgs), _valid)

    return _g_loss, _gen_imgs


def d_forward(_real_imgs, _gen_imgs, _valid, _fake):
    """Discriminator forward function"""
    # Measure discriminator's ability to classify real from generated samples
    real_loss = discriminator_loss(discriminator(_real_imgs), _valid)
    fake_loss = discriminator_loss(discriminator(_gen_imgs), _fake)
    _d_loss = (real_loss + fake_loss) / 2

    return _d_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)

# ----------
#  Training
# ----------

generator.set_train()
discriminator.set_train()

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((imgs.shape[0], 1)))
        fake = ops.stop_gradient(ops.zeros((imgs.shape[0], 1)))

        # Configure input
        real_imgs = Tensor(imgs)

        # -----------------
        #  Train Generator
        # -----------------
        z = ops.randn((imgs.shape[0], opt.latent_dim))

        (g_loss, gen_imgs), g_grads = grad_g(z, valid)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        (d_loss), d_grads = grad_d(real_imgs, ops.stop_gradient(gen_imgs), valid, fake)
        optimizer_D(d_grads)

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] [G loss: {g_loss.asnumpy().item():.4f}]'
        )

        batches_done = epoch * dataset.get_dataset_size() + i

        if batches_done % opt.sample_interval == 0:
            to_image(gen_imgs[:25], os.path.join("images", F'{batches_done}.png'))
