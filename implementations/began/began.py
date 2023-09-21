"""BEGAN Model"""

import argparse
import gzip
import os
import shutil
import urllib.request

import mindspore
import mindspore.common.initializer as init
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
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


class Generator(nn.Cell):
    """GeneratorUNet network"""

    def __init__(self):
        super().__init__(Generator)

        self.init_size = opt.img_size // 4
        self.l1 = nn.SequentialCell(nn.Dense(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.SequentialCell(
            nn.BatchNorm2d(128,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, 128, 3, 1, 'pad', 1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(128, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, 64, 3, 1, 'pad', 1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(64, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, opt.channels, 3, 1, 'pad', 1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        )

    def construct(self, _noise):
        out = self.l1(_noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        _img = self.conv_blocks(out)
        return _img


class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self):
        super().__init__(Discriminator)

        # Upsampling
        self.down = nn.SequentialCell(
            nn.Conv2d(opt.channels, 64, 3,
                      2, 'pad', 1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.ReLU()
        )
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2
        self.fc = nn.SequentialCell(
            nn.Dense(down_dim, 32),
            nn.BatchNorm1d(32, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.ReLU(),
            nn.Dense(32, down_dim),
            nn.BatchNorm1d(down_dim,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.ReLU()
        )
        # Upsampling
        self.up = nn.SequentialCell(nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
                                    nn.Conv2d(64, opt.channels, 3, 1,
                                              'pad', 1, weight_init=init.Normal(0.02, 0.0)))

    def construct(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.shape[0], -1))
        out = self.up(out.view(out.shape[0], 64, self.down_size, self.down_size))
        return out


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


def g_forward(_z):
    """GeneratorUNet forward function"""
    # Generate a batch of images
    _gen_imgs = generator(_z)
    # Loss measures generator's ability to fool the discriminator
    _g_loss = ops.mean(ops.abs(discriminator(_gen_imgs) - _gen_imgs))

    return _g_loss, _gen_imgs


def d_forward(_real_imgs, _gen_imgs):
    """Discriminator forward function"""
    # Measure discriminator's ability to classify real from generated samples
    d_real = discriminator(_real_imgs)
    d_fake = discriminator(_gen_imgs)

    _d_loss_real = ops.mean(ops.abs(d_real - _real_imgs))
    _d_loss_fake = ops.mean(ops.abs(d_fake - _gen_imgs))
    _d_loss = _d_loss_real - k * _d_loss_fake

    return _d_loss, _d_loss_real, _d_loss_fake


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=True)

# ----------
#  Training
# ----------

generator.set_train()
discriminator.set_train()

# BEGAN hyper parameters
gamma = 0.75
lambda_k = 0.001
k = 0.0

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        # Configure input
        real_imgs = Tensor(imgs)

        # -----------------
        #  Train GeneratorUNet
        # -----------------

        z = ops.randn((imgs.shape[0], opt.latent_dim))

        (g_loss, gen_imgs), g_grads = grad_g(z)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        (d_loss, d_loss_real, d_loss_fake), d_grads = grad_d(real_imgs, ops.stop_gradient(gen_imgs))
        optimizer_D(d_grads)

        # ----------------
        # Update weights
        # ----------------

        diff = ops.mean(gamma * d_loss_real - d_loss_fake)

        # Update weight term for fake samples
        k = k + lambda_k * diff.asnumpy().item()
        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        M = d_loss_real + ops.abs(diff)

        # --------------
        # Log Progress
        # --------------

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] [G loss: {g_loss.asnumpy().item():.4f}] '
            f'-- M: {M}, k: {k}'
        )

        batches_done = epoch * dataset.get_dataset_size() + i

        if batches_done % opt.sample_interval == 0:
            to_image(gen_imgs[:25], os.path.join("images", F'{batches_done}.png'))
