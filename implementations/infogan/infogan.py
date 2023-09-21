"""InfoGAN Model"""

import argparse
import gzip
import itertools
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

os.makedirs("images/static/", exist_ok=True)
os.makedirs("images/varying_c1/", exist_ok=True)
os.makedirs("images/varying_c2/", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Tensor(y_cat, dtype=mstype.float32)


class Generator(nn.Cell):
    """GeneratorUNet Network"""

    def __init__(self):
        super().__init__(Generator)
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.SequentialCell(
            nn.Dense(input_dim, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.SequentialCell(
            nn.BatchNorm2d(128,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, 128, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(128, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, 64, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(64, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, opt.channels, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh(),
        )

    def construct(self, noise, _labels, code):
        gen_input = ops.cat((noise, _labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Cell):
    """Discriminator Network"""
    def __init__(self):
        super().__init__(Discriminator)

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [
                nn.Conv2d(in_filters, out_filters, 3,
                          stride=2, pad_mode='pad', padding=1,
                          weight_init=init.Normal(0.02, 0.0)),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8,
                                            gamma_init=init.Normal(0.02, 1.0),
                                            beta_init=init.Constant(0.0), affine=False))
            return block

        self.conv_blocks = nn.SequentialCell(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.SequentialCell(
            nn.Dense(128 * ds_size ** 2, 1)
        )
        self.aux_layer = nn.SequentialCell(
            nn.Dense(128 * ds_size ** 2, opt.n_classes),
            nn.Softmax())
        self.latent_layer = nn.SequentialCell(
            nn.Dense(128 * ds_size ** 2, opt.code_dim)
        )

    def construct(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code


# Loss functions
adversarial_loss = nn.MSELoss()
categorical_loss = nn.CrossEntropyLoss()
continuous_loss = nn.MSELoss()

lambda_cat = 1
lambda_con = 0.1

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
optimizer_info = nn.optim.Adam(
    itertools.chain(generator.trainable_params(), discriminator.trainable_params()),
    learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2
)

# Static generator inputs for sampling
static_z = ops.zeros((opt.n_classes ** 2, opt.latent_dim), dtype=mstype.float32)
static_label = to_categorical(
    np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
)
static_code = ops.zeros((opt.n_classes ** 2, opt.code_dim), dtype=mstype.float32)


def sample_image(_n_row, batches):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = ops.randn((_n_row ** 2, opt.latent_dim), dtype=mstype.float32)
    static_sample = generator(z, static_label, static_code)
    to_image(static_sample, f'images/static/{batches}.png')

    # Get varied c1 and c2
    zeros = np.zeros((_n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, _n_row)[:, np.newaxis], _n_row, 0)
    c1 = Tensor(np.concatenate((c_varied, zeros), -1), dtype=mstype.float32)
    c2 = Tensor(np.concatenate((zeros, c_varied), -1), dtype=mstype.float32)
    sample1 = generator(static_z, static_label, c1)
    sample2 = generator(static_z, static_label, c2)
    to_image(sample1, f'images/varying_c1/{batches}.png')
    to_image(sample2, f'images/varying_c2/{batches}.png')


def g_forward(_imgs, _valid):
    """GeneratorUNet forward function"""
    # Sample noise and labels as generator input
    z = ops.randn((batch_size, opt.latent_dim), dtype=mstype.float32)
    label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
    code_input = ops.uniform((batch_size, opt.code_dim), Tensor(-1.0), Tensor(1.0))

    # Generate a batch of images
    _gen_imgs = generator(z, label_input, code_input)

    # Loss measures generator's ability to fool the discriminator
    validity, _, _ = discriminator(_gen_imgs)
    _g_loss = adversarial_loss(validity, _valid)
    return _g_loss, _gen_imgs


def d_forward(_real_imgs, _gen_imgs, _valid, _fake):
    """Discriminator forward function"""
    # Loss for real images
    real_pred, _, _ = discriminator(_real_imgs)
    d_real_loss = adversarial_loss(real_pred, _valid)

    # Loss for fake images
    fake_pred, _, _ = discriminator(_gen_imgs)
    d_fake_loss = adversarial_loss(fake_pred, _fake)

    # Total discriminator loss
    _d_loss = (d_real_loss + d_fake_loss) / 2

    return _d_loss


def info_forward(_batch_size):
    """Information"""
    # Sample labels
    sampled_labels = np.random.randint(0, opt.n_classes, _batch_size)

    # Ground truth labels
    gt_labels = ops.stop_gradient(Tensor(sampled_labels, dtype=mstype.int32))

    # Sample noise, labels and code as generator input
    z = ops.randn((_batch_size, opt.latent_dim), dtype=mstype.float32)
    label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
    code_input = ops.uniform((_batch_size, opt.code_dim), Tensor(-1.0), Tensor(1.0))

    _gen_imgs = generator(z, label_input, code_input)
    _, pred_label, pred_code = discriminator(_gen_imgs)

    _info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
        pred_code, code_input
    )

    return _info_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)
grad_info = ops.value_and_grad(info_forward, None, optimizer_info.parameters, has_aux=False)

generator.set_train()
discriminator.set_train()

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataset.create_tuple_iterator()):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((batch_size, 1)))
        fake = ops.stop_gradient(ops.zeros((batch_size, 1)))

        # Configure input
        real_imgs = imgs
        labels = to_categorical(labels.asnumpy(), num_columns=opt.n_classes)

        # -----------------
        #  Train GeneratorUNet
        # -----------------

        (g_loss, gen_imgs), g_grads = grad_g(real_imgs, valid)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        (d_loss), d_grads = grad_d(real_imgs, ops.stop_gradient(gen_imgs), valid, fake)
        optimizer_D(d_grads)

        # ------------------
        # Information Loss
        # ------------------

        (info_loss), info_grads = grad_info(batch_size)
        optimizer_info(info_grads)

        # --------------
        # Log Progress
        # --------------

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] [G loss: {g_loss.asnumpy().item():.4f}] '
            f'[info loss: {info_loss.asnumpy().item():.4f}]'
        )
        batches_done = epoch * dataset.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            sample_image(_n_row=10, batches=batches_done)
