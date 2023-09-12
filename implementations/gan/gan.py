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

from implementations.gan.img_utils import to_image

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
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Dense(in_feat, out_feat, weight_init=HeUniform(math.sqrt(5)))]
            if normalize:
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

    def construct(self, x):
        return self.model(x)


class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Dense(int(np.prod(img_shape)), 512, weight_init=HeUniform(math.sqrt(5)))
        self.relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Dense(512, 256, weight_init=HeUniform(math.sqrt(5)))
        self.fc3 = nn.Dense(256, 1, weight_init=HeUniform(math.sqrt(5)))
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


generator = Generator()
discriminator = Discriminator()

# 损失函数与优化器
criterion = nn.BCELoss()
D_Optim = nn.optim.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
G_Optim = nn.optim.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)


# 生成器正向传播
def g_forward(z):
    img = generator(z)
    img = img.view(img.shape[0], *img_shape)
    img_flat = img.view(img.shape[0], -1)
    dis = discriminator(img_flat)
    _g_loss = criterion(dis, valid)
    return _g_loss, img


# 判别器正向传播
def d_forward(img, gen_img, _valid, _fake):
    img_flat = img.view(img.shape[0], -1)
    gen_flat = gen_img.view(img.shape[0], -1)
    # _real_score = discriminator(img)
    # _fake_score = discriminator(img_flat)
    real_loss = criterion(discriminator(img_flat), _valid)
    fake_loss = criterion(discriminator(gen_flat), _fake)
    _d_loss = real_loss + fake_loss
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
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        generator.set_train()
        discriminator.set_train()
        valid = ops.ones((imgs.shape[0], 1))
        fake = ops.zeros((imgs.shape[0], 1))
        ops.stop_gradient(valid)
        ops.stop_gradient(fake)

        real_imgs = Tensor(imgs)

        z = ops.randn((imgs.shape[0], opt.latent_dim))

        (g_loss, gen_images), g_grads = grad_g(z)
        G_Optim(g_grads)

        (d_loss), d_grads = grad_d(imgs, gen_images, valid, fake)
        D_Optim(d_grads)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, dataset.get_dataset_size(), d_loss.asnumpy().item(), g_loss.asnumpy().item())
        )

        batches_done = epoch * dataset.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            to_image(gen_images[:25], os.path.join("images", F'{batches_done}.png'))
