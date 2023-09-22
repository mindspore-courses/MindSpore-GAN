"""ACGAN Model"""

import argparse
import os

import mindspore
import mindspore.common.initializer as init
import numpy as np
from mindspore import Tensor, ops
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import transforms

from img_utils import to_image

file_path = "../../data/MNIST/"

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


class Generator(nn.Cell):
    """Generator Network"""

    def __init__(self):
        super().__init__(Generator)

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.SequentialCell(nn.Dense(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.SequentialCell(
            nn.BatchNorm2d(128,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0)),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, 128, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(128, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0)),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, 64, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(64, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, opt.channels, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh(),
        )

    def construct(self, _noise, _labels):
        gen_input = ops.mul(self.label_emb(_labels), _noise)
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
                nn.Conv2d(in_filters, out_filters,
                          3, 2, 'pad', 1,
                          weight_init=init.Normal(0.02, 0.0)),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8,
                                            gamma_init=init.Normal(0.02, 1.0),
                                            beta_init=init.Constant(0.0)))
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
            nn.Dense(128 * ds_size ** 2, opt.n_classes),
            nn.Softmax())

    def construct(self, _img):
        out = self.conv_blocks(_img)
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
generator.update_parameters_name('generator')
discriminator.update_parameters_name('discriminator')
generator.set_train()
discriminator.set_train()

optimizer_G = nn.optim.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)

optimizer_G.update_parameters_name('optimizer_G')
optimizer_D.update_parameters_name('optimizer_D')


def sample_image(n_row, batches):
    """Saves a grid of generated digits"""
    # Sample noise
    _z = ops.randn((n_row ** 2, opt.latent_dim))
    # Get labels ranging from 0 to n_classes for n rows
    _labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    _labels = Tensor(_labels)
    _gen_imgs = generator(_z, _labels)
    to_image(_gen_imgs, os.path.join("images", F'{batches}.png'))


def g_forward(_z, _gen_labels):
    """Generator forward function"""
    _gen_imgs = generator(_z, _gen_labels)
    # Loss measures generator's ability to fool the discriminator
    validity, pred_label = discriminator(_gen_imgs)
    _g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, _gen_labels))
    return _g_loss, _gen_imgs


def d_forward(_real_imgs, _gen_imgs, _labels, _gen_labels, _valid, _fake):
    """Discriminator forward function"""
    # Loss for real images
    real_pred, _real_aux = discriminator(_real_imgs)
    d_real_loss = (adversarial_loss(real_pred, _valid) + auxiliary_loss(_real_aux, _labels)) / 2

    # Loss for fake images
    fake_pred, _fake_aux = discriminator(_gen_imgs)
    d_fake_loss = (adversarial_loss(fake_pred, _fake) + auxiliary_loss(_fake_aux, gen_labels)) / 2

    # Total discriminator loss
    _d_loss = (d_real_loss + d_fake_loss) / 2
    return _d_loss, _real_aux, _fake_aux


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

grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=True)

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataset.create_tuple_iterator()):
        batch_size = imgs.shape[0]
        valid = ops.stop_gradient(ops.ones((batch_size, 1)))
        fake = ops.stop_gradient(ops.zeros((batch_size, 1)))

        real_imgs = Tensor(imgs, dtype=mstype.float32)
        labels = Tensor(labels, dtype=mstype.int32)

        z = ops.randn((batch_size, opt.latent_dim))
        gen_labels = ops.randint(0, opt.n_classes, (batch_size,), dtype=mstype.int32)

        # ---------------------
        #  Train Generator
        # ---------------------

        (g_loss, gen_imgs), g_grads = grad_g(z, gen_labels)
        optimizer_G(g_grads)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        (d_loss, real_aux, fake_aux), d_grads = grad_d(real_imgs, ops.stop_gradient(gen_imgs),
                                                       labels, gen_labels, valid, fake)

        pred = np.concatenate([real_aux.asnumpy(), fake_aux.asnumpy()], axis=0)
        gt = np.concatenate([labels.asnumpy(), gen_labels.asnumpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        optimizer_D(d_grads)

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{dataset.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f},acc:{100 * d_acc:.2f}%] [G loss: {g_loss.asnumpy().item():.4f}]'
        )
        batches_done = epoch * dataset.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            sample_image(10, batches_done)
