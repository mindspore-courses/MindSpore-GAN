"""COGAN Model"""

import argparse
import gzip
import os
import shutil
import urllib.request

import mindspore
import mindspore.common.initializer as init
from mindspore import nn
from mindspore import ops
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import transforms

from img_utils import to_image
from mnistm import MNISTM

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
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


class CoupledGenerators(nn.Cell):
    """COGAN Generators"""

    def __init__(self):
        super().__init__(CoupledGenerators)

        self.init_size = opt.img_size // 4
        self.fc = nn.SequentialCell(nn.Dense(opt.latent_dim, 128 * self.init_size ** 2))

        self.shared_conv = nn.SequentialCell(
            nn.BatchNorm2d(128,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, 128, 3, stride=1,
                      pad_mode='pad', padding=1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(128, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
        )
        self.G1 = nn.SequentialCell(
            nn.Conv2d(128, 64, 3, stride=1,
                      pad_mode='pad', padding=1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(64, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, opt.channels, 3, stride=1,
                      pad_mode='pad', padding=1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        )
        self.G2 = nn.SequentialCell(
            nn.Conv2d(128, 64, 3, stride=1,
                      pad_mode='pad', padding=1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(64, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, opt.channels, 3, stride=1,
                      pad_mode='pad', padding=1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        )

    def construct(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)
        img1 = self.G1(img_emb)
        img2 = self.G2(img_emb)
        return img1, img2


class CoupledDiscriminators(nn.Cell):
    """COGAN Discriminator"""

    def __init__(self):
        super().__init__(CoupledDiscriminators)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2,
                               pad_mode='pad', padding=1, has_bias=False,
                               weight_init=init.Normal(0.02, 0.0))]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8,
                                            gamma_init=init.Normal(0.02, 1.0),
                                            beta_init=init.Constant(0.0), affine=False))
            block.extend([nn.LeakyReLU(0.2), nn.Dropout2d(0.25)])
            return block

        self.shared_conv = nn.SequentialCell(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.D1 = nn.Dense(128 * ds_size ** 2, 1)
        self.D2 = nn.Dense(128 * ds_size ** 2, 1)

    def construct(self, img1, img2):
        # Determine validity of first image
        out = self.shared_conv(img1)
        out = out.view(out.shape[0], -1)
        validity1 = self.D1(out)
        # Determine validity of second image
        out = self.shared_conv(img2)
        out = out.view(out.shape[0], -1)
        validity2 = self.D2(out)

        return validity1, validity2


# Loss function
adversarial_loss = nn.MSELoss()

# Initialize models
coupled_generators = CoupledGenerators()
coupled_discriminators = CoupledDiscriminators()

transform = [
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5],is_hwc=False)
]


os.makedirs("../../data/MNIST-M", exist_ok=True)

dataset1 = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='train',
    shuffle=True
).map(operations=transform, input_columns="image").batch(opt.batch_size)

dataset2 = mindspore.dataset.GeneratorDataset(
    source=MNISTM(
        root='../../data/MNIST-M',
        mnist_root='../../data/MNIST',
        transform=Compose(
            [
                transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), is_hwc=False),
            ]
        )
    ),
    shuffle=True,
    column_names=["image", "target"]
).batch(opt.batch_size)

# Optimizers
optimizer_G = nn.optim.Adam(coupled_generators.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.optim.Adam(coupled_discriminators.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)


def g_forward(_batch_size, _valid):
    """GeneratorUNet forward function"""
    # Sample noise as generator input
    z = ops.randn((_batch_size, opt.latent_dim))

    # Generate a batch of images
    _gen_imgs1, _gen_imgs2 = coupled_generators(z)
    # Determine validity of generated images
    validity1, validity2 = coupled_discriminators(_gen_imgs1, _gen_imgs2)

    _g_loss = (adversarial_loss(validity1, _valid) + adversarial_loss(validity2, _valid)) / 2

    return _g_loss, _gen_imgs1, _gen_imgs2


def d_forward(_imgs1, _imgs2, _gen_imgs1, _gen_imgs2, _valid, _fake):
    """Discriminator forward function"""
    # Determine validity of real and generated images
    validity1_real, validity2_real = coupled_discriminators(_imgs1, _imgs2)
    validity1_fake, validity2_fake = coupled_discriminators(_gen_imgs1, _gen_imgs2)

    _d_loss = (
                      adversarial_loss(validity1_real, _valid)
                      + adversarial_loss(validity1_fake, _fake)
                      + adversarial_loss(validity2_real, _valid)
                      + adversarial_loss(validity2_fake, _fake)
              ) / 4

    return _d_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)

coupled_generators.set_train()
coupled_discriminators.set_train()

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, ((imgs1, _), (imgs2, _)) in enumerate(zip(dataset1.create_tuple_iterator(),
                                                     dataset2.create_tuple_iterator())):
        batch_size = imgs1.shape[0]

        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((batch_size, 1)))
        fake = ops.stop_gradient(ops.zeros((batch_size, 1)))

        # Configure input
        imgs1 = imgs1.broadcast_to((imgs1.shape[0], 3, opt.img_size, opt.img_size))

        # ------------------
        #  Train Generators
        # ------------------

        (g_loss, gen_imgs1, gen_imgs2), g_grads = grad_g(batch_size, valid)
        optimizer_G(g_grads)

        # ----------------------
        #  Train Discriminators
        # ----------------------

        (d_loss), d_grads = grad_d(imgs1, imgs2, ops.stop_gradient(gen_imgs1),
                                   ops.stop_gradient(gen_imgs2), valid, fake)
        optimizer_D(d_grads)

        # --------------
        # Log Progress
        # --------------

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{dataset1.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] [G loss: {g_loss.asnumpy().item():.4f}]'
        )
        batches_done = epoch * dataset1.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            gen_imgs = ops.cat((gen_imgs1, gen_imgs2), 0)
            to_image(gen_imgs, os.path.join("images", F'{batches_done}.png'))
