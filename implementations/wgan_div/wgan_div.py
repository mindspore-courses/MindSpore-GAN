"""WGAN DIV model"""

import argparse
import gzip
import os
import shutil
import urllib.request

import mindspore
import numpy as np
from matplotlib import pyplot as plt
from mindspore import Tensor, ops
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import transforms

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
parser.add_argument("--n_epochs",
                    type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size",
                    type=int, default=64, help="size of the batches")
parser.add_argument("--lr",
                    type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1",
                    type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2",
                    type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu",
                    type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim",
                    type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size",
                    type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels",
                    type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic",
                    type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value",
                    type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval",
                    type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

image_path = "./images"


def save_imgs(_gen_imgs, idx):
    """Save the generated test image."""
    for j in range(_gen_imgs.shape[0]):
        plt.subplot(5, 5, j + 1)
        plt.imshow(_gen_imgs[j, 0, :, :] / 2 + 0.5, cmap="gray")
        plt.axis("off")
    plt.savefig(f'image_path/test_{idx}.png"')


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
        img = self.model(_z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self):
        super().__init__(Discriminator)

        self.model = nn.SequentialCell(
            nn.Dense(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dense(256, 1)
        )

    def construct(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


k = 2
p = 6

generator = Generator()
discriminator = Discriminator()
generator.update_parameters_name('generator')
discriminator.update_parameters_name('discriminator')
generator.set_train()
discriminator.set_train()

G_Optim = nn.optim.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
D_Optim = nn.optim.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)

G_Optim.update_parameters_name('G_Optim')
D_Optim.update_parameters_name('D_Optim')


def g_forward(_z):
    """Generator forward function"""
    fake_imgs = generator(_z)
    # Loss measures generator's ability to fool the discriminator
    # Train on fake images
    fake_validity = discriminator(fake_imgs)
    g_loss = -ops.mean(fake_validity)
    return g_loss, fake_imgs


def d_forward(_real_imgs):
    """Discriminator forward function"""
    _z = ops.randn((imgs.shape[0], opt.latent_dim))
    fake_imgs = generator(_z)
    # Real images
    real_validity = discriminator(_real_imgs)
    # Fake images
    fake_validity = discriminator(fake_imgs)

    # Compute W-div gradient penalty
    real_grad_out = ops.ones((_real_imgs.shape[0], 1))
    ops.stop_gradient(real_grad_out)
    real_grad_fn = ops.grad(discriminator)
    real_gradients = real_grad_fn(_real_imgs)
    real_grad_norm = real_gradients.view(real_gradients.shape[0], -1).pow(2).sum(1) ** (p / 2)

    fake_grad_out = ops.ones((_real_imgs.shape[0], 1))
    ops.stop_gradient(fake_grad_out)
    fake_grad_fn = ops.grad(discriminator)
    fake_gradients = fake_grad_fn(fake_imgs)
    fake_grad_norm = fake_gradients.view(fake_gradients.shape[0], -1).pow(2).sum(1) ** (p / 2)

    div_gp = ops.mean(real_grad_norm + fake_grad_norm) * k / 2
    # Adversarial loss
    d_loss = -ops.mean(real_validity) + ops.mean(fake_validity) + div_gp

    return d_loss, _z


transform = [
    transforms.Rescale(1.0 / 255.0, 0),
    transforms.Resize(opt.img_size),
    transforms.Normalize([0.5], [0.5]),
    transforms.HWC2CHW()
]

dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='train',
    shuffle=True
).map(operations=transform, input_columns="image").batch(opt.batch_size)

grad_g = ops.value_and_grad(g_forward, None, G_Optim.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, D_Optim.parameters, has_aux=True)

batches_done = 0

generator.update_parameters_name('generator')
discriminator.update_parameters_name('discriminator')
discriminator.set_train()
generator.set_train()

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        real_imgs = Tensor(imgs, dtype=mstype.float32)

        (loss_D, z), D_grads = grad_d(real_imgs)
        D_Optim(D_grads)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            (loss_G, gen_imgs), G_grads = grad_g(z)
            G_Optim(G_grads)

            print(
                f'[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{dataset.get_dataset_size()}] '
                f'[D loss: {loss_D.asnumpy().item()}] [G loss: {loss_G.asnumpy().item()}]'
            )

            if batches_done % opt.sample_interval == 0:
                save_imgs(gen_imgs[:25].asnumpy(), batches_done)
                # to_image(gen_imgs[:25], os.path.join("images", F'{batches_done}.png'))
            batches_done += opt.n_critic
