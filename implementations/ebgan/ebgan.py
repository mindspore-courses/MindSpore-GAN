"""EBGAN Model"""

import argparse
import gzip
import os
import shutil
import urllib.request

import mindspore
import mindspore.common.initializer as init
from mindspore import nn
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
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


class Generator(nn.Cell):
    """Generator Network"""

    def __init__(self):
        super().__init__(Generator)

        self.init_size = opt.img_size // 4
        self.l1 = nn.SequentialCell(
            nn.Dense(opt.latent_dim, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.SequentialCell(
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

    def construct(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self):
        super().__init__(Discriminator)

        # Upsampling
        self.down = nn.SequentialCell(
            nn.Conv2d(opt.channels, 64, 3,
                      stride=2, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.ReLU()
        )
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2

        self.embedding = nn.Dense(down_dim, 32)

        self.fc = nn.SequentialCell(
            nn.BatchNorm1d(32, 0.8, affine=False),
            nn.ReLU(),
            nn.Dense(32, down_dim),
            nn.BatchNorm1d(down_dim, affine=False),
            nn.ReLU(),
        )
        # Upsampling
        self.up = nn.SequentialCell(
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(64, opt.channels, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0))
        )

    def construct(self, img):
        out = self.down(img)
        embedding = self.embedding(out.view(out.shape[0], -1))
        out = self.fc(embedding)
        out = self.up(out.view(out.shape[0], 64, self.down_size, self.down_size))
        return out, embedding


# Reconstruction loss of AE
pixelwise_loss = nn.MSELoss()

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


def pullaway_loss(embeddings):
    """Pull away loss"""
    norm = ops.sqrt(ops.sum(embeddings ** 2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = ops.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.shape[0]
    loss_pt = (ops.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt


def g_forward(_imgs):
    """Generator forward function"""
    # Sample noise as generator input
    z = ops.randn((_imgs.shape[0], opt.latent_dim), dtype=mstype.float32)

    # Generate a batch of images
    _gen_imgs = generator(z)
    recon_imgs, img_embeddings = discriminator(_gen_imgs)

    # Loss measures generator's ability to fool the discriminator
    _g_loss = pixelwise_loss(recon_imgs, _gen_imgs) + lambda_pt * pullaway_loss(img_embeddings)

    return _g_loss, _gen_imgs


def d_forward(_real_imgs, _gen_imgs):
    """Discriminator forward function"""
    # Measure discriminator's ability to classify real from generated samples
    real_recon, _ = discriminator(_real_imgs)
    fake_recon, _ = discriminator(_gen_imgs)

    d_loss_real = pixelwise_loss(real_recon, _real_imgs)
    d_loss_fake = pixelwise_loss(fake_recon, _gen_imgs)

    _d_loss = d_loss_real
    if (margin - d_loss_fake).asnumpy().item() > 0:
        _d_loss += margin - d_loss_fake

    return _d_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)

generator.set_train()
discriminator.set_train()

# ----------
#  Training
# ----------

# BEGAN hyper parameters
lambda_pt = 0.1
margin = max(1, opt.batch_size / 64.0)

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        # Configure input
        real_imgs = imgs

        # -----------------
        #  Train Generator
        # -----------------

        (g_loss, gen_imgs), g_grads = grad_g(real_imgs)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        (d_loss), d_grads = grad_d(real_imgs, gen_imgs)
        optimizer_D(d_grads)

        # --------------
        # Log Progress
        # --------------

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] [G loss: {g_loss.asnumpy().item():.4f}]'
        )
        batches_done = epoch * dataset.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            to_image(gen_imgs[:25], os.path.join("images", F'{batches_done}.png'))
