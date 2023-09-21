"""RelativisticGAN Model"""

import argparse
import os

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import transforms

from img_utils import to_image

file_path = "../../data/MNIST/"

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--rel_avg_gan", action="store_true", help="relativistic average GAN instead of standard")
opt = parser.parse_args()
print(opt)


class Generator(nn.Cell):
    """GeneratorUNet Network"""

    def __init__(self):
        super().__init__(Generator)

        self.init_size = opt.img_size // 4
        self.l1 = nn.SequentialCell(nn.Dense(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.SequentialCell(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, 128, 3, stride=1,
                      pad_mode='pad', padding=1),
            nn.BatchNorm2d(128, 0.8, affine=False),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, 64, 3, stride=1,
                      pad_mode='pad', padding=1),
            nn.BatchNorm2d(64, 0.8, affine=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, opt.channels, 3, stride=1,
                      pad_mode='pad', padding=1),
            nn.Tanh()
        )

    def construct(self, _z):
        out = self.l1(_z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self):
        super().__init__(Discriminator)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2,
                          'pad', 1),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8, affine=False))
            return block

        self.model = nn.SequentialCell(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.SequentialCell(nn.Dense(128 * ds_size ** 2, 1))

    def construct(self, _img):
        out = self.model(_img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = nn.BCEWithLogitsLoss()

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


def g_forward(_real_imgs, _valid):
    """GeneratorUNet forward function"""
    # Generate a batch of images
    z = ops.randn((_real_imgs.shape[0], opt.latent_dim), dtype=mstype.float32)
    _gen_imgs = generator(z)

    real_pred = ops.stop_gradient(discriminator(_real_imgs))
    fake_pred = discriminator(_gen_imgs)

    if opt.rel_avg_gan:
        _g_loss = adversarial_loss(fake_pred - real_pred.mean(0, keep_dims=True), _valid)
    else:
        _g_loss = adversarial_loss(fake_pred - real_pred, _valid)

    # Loss measures generator's ability to fool the discriminator
    _g_loss = adversarial_loss(discriminator(_gen_imgs), _valid)

    return _g_loss, _gen_imgs


def d_forward(_real_imgs, _gen_imgs, _valid, _fake):
    """Discriminator forward function"""

    # Predict validity
    real_pred = discriminator(_real_imgs)
    fake_pred = discriminator(_gen_imgs)

    if opt.rel_avg_gan:
        real_loss = adversarial_loss(real_pred - fake_pred.mean(0, keep_dims=True), _valid)
        fake_loss = adversarial_loss(fake_pred - real_pred.mean(0, keep_dims=True), _fake)
    else:
        real_loss = adversarial_loss(real_pred - fake_pred, _valid)
        fake_loss = adversarial_loss(fake_pred - real_pred, _fake)

    _d_loss = (real_loss + fake_loss) / 2

    return _d_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)

# ----------
#  Training
# ----------

generator.set_train()
discriminator.set_train()

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):

        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((imgs.shape[0], 1)))
        fake = ops.stop_gradient(ops.zeros((imgs.shape[0], 1)))

        # Configure input
        real_imgs = imgs

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

        # --------------
        # Log Progress
        # --------------

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] '
            f'[Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] '
            f'[G loss: {g_loss.asnumpy().item():.4f}]'
        )
        batches_done = epoch * dataset.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            to_image(gen_imgs[:25], os.path.join("images", F'{batches_done}.png'))
