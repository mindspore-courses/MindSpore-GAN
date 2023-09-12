import argparse
import gzip
import math
import os
import shutil
import urllib
from urllib import request

import numpy as np
import mindspore
from mindspore import Tensor, ops
from mindspore import nn
from mindspore.common.initializer import HeUniform
from mindspore.dataset.vision import transforms

from implementations.aae.img_utils import to_image

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
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


def reparameterization(mu, logvar):
    std = ops.exp(logvar / 2)
    sampled_z = ops.randn((mu.shape[0], opt.latent_dim))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.SequentialCell(
            nn.Dense(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, ),
            nn.Dense(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )

        self.mu = nn.Dense(512, opt.latent_dim)
        self.logvar = nn.Dense(512, opt.latent_dim)

    def construct(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.SequentialCell(
            nn.Dense(opt.latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dense(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def construct(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.SequentialCell(
            nn.Dense(opt.latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dense(256, 1),
            nn.Sigmoid(),
        )

    def construct(self, z):
        validity = self.model(z)
        return validity


encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()
encoder.update_parameters_name("encoder")
decoder.update_parameters_name("decoder")
discriminator.update_parameters_name("discriminator")

# Use binary cross-entropy loss
adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.L1Loss()

params = list(encoder.trainable_params()) + list(decoder.trainable_params())
optimizer_G = nn.optim.Adam(params, learning_rate=opt.lr,
                            beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), learning_rate=opt.lr,
                            beta1=opt.b1, beta2=opt.b2)


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = ops.randn((n_row ** 2, opt.latent_dim))
    gen_imgs = decoder(z)
    to_image(gen_imgs, os.path.join("images", F'{batches_done}.png'))


# 生成器正向传播
def g_forward(real_imgs):
    encoded_imgs = encoder(real_imgs)
    decoded_imgs = decoder(encoded_imgs)
    g_loss = (0.001 * adversarial_loss(discriminator(encoded_imgs), valid)
              + 0.999 * pixelwise_loss(decoded_imgs, real_imgs))
    return g_loss, encoded_imgs


# 判别器正向传播
def d_forward(encoded_imgs, valid, fake):
    z = ops.randn((imgs.shape[0], opt.latent_dim))
    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(z), valid)
    fake_loss = adversarial_loss(discriminator(encoded_imgs), fake)
    d_loss = 0.5 * (real_loss + fake_loss)
    return d_loss


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

grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)

# 训练
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        encoder.set_train()
        decoder.set_train()
        discriminator.set_train()
        valid = ops.ones((imgs.shape[0], 1))
        fake = ops.zeros((imgs.shape[0], 1))
        ops.stop_gradient(valid)
        ops.stop_gradient(fake)

        real_imgs = Tensor(imgs)

        # -----------------
        #  Train Generator
        # -----------------

        (g_loss, encoded_imgs), g_grads = grad_g(real_imgs)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        (d_loss), d_grads = grad_d(ops.stop_gradient(encoded_imgs), valid, fake)
        optimizer_D(d_grads)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, dataset.get_dataset_size(), d_loss.asnumpy().item(), g_loss.asnumpy().item())
        )

        batches_done = epoch * dataset.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            sample_image(10, batches_done=batches_done)
