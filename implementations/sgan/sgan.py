"""SGAN Model"""

import argparse
import gzip
import os
import shutil
import urllib.request

import mindspore
import mindspore.common.initializer as init
import numpy as np
from mindspore import nn, Tensor
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
parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)


class Generator(nn.Cell):
    """Generator Network"""

    def __init__(self):
        super().__init__(Generator)

        self.label_emb = nn.Embedding(opt.num_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.SequentialCell(
            nn.Dense(opt.latent_dim, 128 * self.init_size ** 2)
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
            nn.Tanh()
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
            nn.Dense(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
        self.aux_layer = nn.SequentialCell(
            nn.Dense(128 * ds_size ** 2, opt.num_classes + 1),
            nn.Softmax()
        )

    def construct(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


# Loss functions
adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()

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


def g_forward(_batch_size, _valid):
    """Generator forward function"""
    # Sample noise and labels as generator input
    z = ops.randn((_batch_size, opt.latent_dim), dtype=mstype.float32)

    # Generate a batch of images
    _gen_imgs = generator(z)

    # Loss measures generator's ability to fool the discriminator
    validity, _ = discriminator(_gen_imgs)
    _g_loss = adversarial_loss(validity, _valid)

    return _g_loss, _gen_imgs


def d_forward(_real_imgs, _gen_imgs, _labels, _fake_aux_gt, _valid, _fake):
    """Discriminator forward function"""
    # Loss for real images
    real_pred, real_aux = discriminator(_real_imgs)
    d_real_loss = (adversarial_loss(real_pred, _valid) + auxiliary_loss(real_aux, _labels)) / 2

    # Loss for fake images
    fake_pred, fake_aux = discriminator(_gen_imgs)
    d_fake_loss = (adversarial_loss(fake_pred, _fake) + auxiliary_loss(fake_aux, _fake_aux_gt)) / 2

    # Total discriminator loss
    _d_loss = (d_real_loss + d_fake_loss) / 2

    # Calculate discriminator accuracy
    pred = np.concatenate([real_aux.asnumpy(), fake_aux.asnumpy()], axis=0)
    gt = np.concatenate([_labels.asnumpy(), _fake_aux_gt.asnumpy()], axis=0)
    _d_acc = np.mean(np.argmax(pred, axis=1) == gt)

    return _d_loss, _d_acc


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=True)

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
        fake_aux_gt = ops.stop_gradient(ops.fill(mstype.int32, Tensor(batch_size), opt.num_classes))

        # Configure input
        real_imgs = imgs
        labels = Tensor(labels, dtype=mstype.int32)

        # -----------------
        #  Train Generator
        # -----------------

        (g_loss, gen_imgs), g_grads = grad_g(batch_size, valid)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        (d_loss, d_acc), d_grads = grad_d(real_imgs, ops.stop_gradient(gen_imgs),
                                          labels, fake_aux_gt, valid, fake)
        optimizer_D(d_grads)

        # --------------
        # Log Progress
        # --------------

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] '
            f'[Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}, '
            f'acc: {100 * d_acc}%] '
            f'[G loss: {g_loss.asnumpy().item():.4f}]'
        )
        batches_done = epoch * dataset.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            to_image(gen_imgs[:25], os.path.join("images", F'{batches_done}.png'))
