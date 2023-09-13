"""WGAN Model"""

import argparse
import gzip
import os
import shutil
import urllib.request

import mindspore
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
                    type=float, default=0.00005, help="learning rate")
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


generator = Generator()
discriminator = Discriminator()

G_Optim = nn.optim.RMSProp(generator.trainable_params(), learning_rate=opt.lr, decay=0.99, epsilon=1e-08)
D_Optim = nn.optim.RMSProp(discriminator.trainable_params(), learning_rate=opt.lr, decay=0.99, epsilon=1e-08)


def g_forward(_z):
    """Generator forward function"""
    _gen_imgs = generator(_z)
    _loss_G = -ops.mean(discriminator(_gen_imgs))
    return _loss_G, _gen_imgs


# 判别器正向传播
def d_forward(_real_imgs, _z):
    """Discriminator forward function"""
    fake_imgs = ops.stop_gradient(generator(_z))
    _loss_D = -ops.mean(discriminator(_real_imgs)) + ops.mean(discriminator(fake_imgs))

    return _loss_D


transform = [
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
]

dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='train',
    shuffle=True
).map(operations=transform, input_columns="image").batch(opt.batch_size)

grad_g = ops.value_and_grad(g_forward, None, G_Optim.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, D_Optim.parameters, has_aux=False)

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        # Train Discriminator
        discriminator.set_train()
        generator.set_train(False)
        real_imgs = Tensor(imgs, dtype=mstype.float32)

        z = ops.randn((imgs.shape[0], opt.latent_dim))

        (loss_D), D_grads = grad_d(real_imgs, z)
        D_Optim(D_grads)

        # Clip weights of discriminator
        for p in discriminator.get_parameters():
            ops.assign(p, ops.clamp(p, -opt.clip_value, opt.clip_value))

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            generator.set_train()
            (loss_G, gen_imgs), G_grads = grad_g(z)
            G_Optim(G_grads)

            print(
                f'[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{dataset.get_dataset_size()}] '
                f'[D loss: {loss_D.asnumpy().item()}] [G loss: {loss_G.asnumpy().item()}]'
            )

        if batches_done % opt.sample_interval == 0:
            to_image(gen_imgs[:25], os.path.join("images", F'{batches_done}.png'))
        batches_done += 1
