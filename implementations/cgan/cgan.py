"""CGAN Model"""

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
                    type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1",
                    type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2",
                    type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu",
                    type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim",
                    type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes",
                    type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size",
                    type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels",
                    type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval",
                    type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


class Generator(nn.Cell):
    """Generator Network"""

    def __init__(self):
        super().__init__(Generator)

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Dense(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.SequentialCell(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Dense(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def construct(self, noise, _labels):
        gen_input = ops.cat((self.label_emb(_labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self):
        super().__init__(Discriminator)

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.SequentialCell(
            nn.Dense(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 1)
        )

    def construct(self, img, _labels):
        d_in = ops.cat((img.view(img.shape[0], -1),
                        self.label_emb(_labels)), -1)
        validity = self.model(d_in)
        return validity


# 损失函数
adversarial_loss = nn.MSELoss()

generator = Generator()
discriminator = Discriminator()

G_Optim = nn.optim.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
D_Optim = nn.optim.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)


def g_forward(_z, _gen_labels, _valid):
    """Generator forward function"""
    _gen_imgs = generator(_z, _gen_labels)
    validity = discriminator(_gen_imgs, _gen_labels)
    _g_loss = adversarial_loss(validity, _valid)
    return _g_loss, _gen_imgs


def d_forward(_real_imgs, _gen_imgs, _labels, _gen_labels, _valid, _fake):
    """Discriminator forward function"""
    validity_real = discriminator(_real_imgs, _labels)
    d_real_loss = adversarial_loss(validity_real, _valid)

    validity_fake = discriminator(_gen_imgs, _gen_labels)
    d_fake_loss = adversarial_loss(validity_fake, _fake)

    _d_loss = (d_real_loss + d_fake_loss) / 2

    return _d_loss


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

grad_g = ops.value_and_grad(g_forward, None, G_Optim.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, D_Optim.parameters, has_aux=False)

# 训练
for epoch in range(opt.n_epochs):
    generator.set_train()
    discriminator.set_train()
    for i, (imgs, labels) in enumerate(dataset.create_tuple_iterator()):
        batch_size = imgs.shape[0]

        valid = ops.ones((batch_size, 1))
        fake = ops.zeros((batch_size, 1))
        ops.stop_gradient(valid)
        ops.stop_gradient(fake)

        real_imgs = Tensor(imgs, dtype=mstype.float32)
        labels = Tensor(labels, dtype=mstype.int64)

        z = ops.randn((batch_size, opt.latent_dim))

        # gen_labels = Tensor(Tensor(np.random.randint(0, opt.n_classes, batch_size)), dtype=mstype.int64)
        gen_labels = ops.randint(0, opt.n_classes, (batch_size,))
        (g_loss, gen_imgs), g_grads = grad_g(z, gen_labels, valid)
        G_Optim(g_grads)

        (d_loss), d_grads = grad_d(real_imgs, gen_imgs, labels,
                                   gen_labels, valid, fake)
        D_Optim(d_grads)

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item()}] [G loss: {g_loss.asnumpy().item()}]'
        )

        batches_done = epoch * dataset.get_dataset_size() + i

        z = ops.randn((10 ** 2, opt.latent_dim))

        labels = np.array([num for _ in range(10) for num in range(10)])
        labels = Tensor(Tensor(labels), dtype=mstype.int64)
        gen_imgs = generator(z, labels)
        if batches_done % opt.sample_interval == 0:
            to_image(gen_imgs, os.path.join("images", F'{batches_done}.png'))
