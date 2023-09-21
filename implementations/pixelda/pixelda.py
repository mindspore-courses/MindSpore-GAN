"""PixelDA Model"""

import argparse
import gzip
import itertools
import os
import shutil
import urllib.request

import mindspore
import mindspore.common.initializer as init
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore.common import dtype as mstype
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
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the noise input")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes in the dataset")
parser.add_argument("--sample_interval", type=int, default=300, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / 2 ** 4)
patch = (1, patch, patch)


class ResidualBlock(nn.Cell):
    """Residual block"""

    def __init__(self, in_features=64):
        super().__init__(ResidualBlock)

        self.block = nn.SequentialCell(
            nn.Conv2d(in_features, in_features, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(in_features,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(in_features,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False)
        )

    def construct(self, x):
        return x + self.block(x)


class Generator(nn.Cell):
    """GeneratorUNet Network"""

    def __init__(self):
        super().__init__(Generator)

        # Fully-connected layer which constructs image channel shaped output from noise
        self.fc = nn.Dense(opt.latent_dim, opt.channels * opt.img_size ** 2)

        self.l1 = nn.SequentialCell(
            nn.Conv2d(opt.channels * 2, 64, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.ReLU()
        )

        resblocks = []
        for _ in range(opt.n_residual_blocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.SequentialCell(*resblocks)

        self.l2 = nn.SequentialCell(
            nn.Conv2d(64, opt.channels, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        )

    def construct(self, img, z):
        gen_input = ops.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)

        return img_


class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self):
        super().__init__(Discriminator)

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 3,
                                stride=2, pad_mode='pad', padding=1,
                                weight_init=init.Normal(0.02, 0.0)),
                      nn.LeakyReLU(0.2)
                      ]
            if normalization:
                layers.append(nn.BatchNorm2d(out_features))
            return layers

        self.model = nn.SequentialCell(
            *block(opt.channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3,
                      stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0))
        )

    def construct(self, img):
        validity = self.model(img)

        return validity


class Classifier(nn.Cell):
    """Classifier Network"""
    def __init__(self):
        super().__init__(Classifier)

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [
                nn.Conv2d(in_features, out_features, 3, stride=2,
                          pad_mode='pad', padding=1,
                          weight_init=init.Normal(0.02, 0.0)),
                nn.LeakyReLU(0.2)
            ]
            if normalization:
                layers.append(nn.BatchNorm2d(out_features,
                                             gamma_init=init.Normal(0.02, 1.0),
                                             beta_init=init.Constant(0.0), affine=False))
            return layers

        self.model = nn.SequentialCell(
            *block(opt.channels, 64, normalization=False), *block(64, 128), *block(128, 256), *block(256, 512)
        )

        input_size = opt.img_size // 2 ** 4
        self.output_layer = nn.SequentialCell(
            nn.Dense(512 * input_size ** 2, opt.n_classes),
            nn.Softmax()
        )

    def construct(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.shape[0], -1)
        label = self.output_layer(feature_repr)
        return label


# Loss function
adversarial_loss = nn.MSELoss()
task_loss = nn.CrossEntropyLoss()

# Loss weights
lambda_adv = 1
lambda_task = 0.1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
classifier = Classifier()

transform = [
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], is_hwc=False)
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
optimizer_G = nn.optim.Adam(itertools.chain(generator.trainable_params(), classifier.trainable_params()),
                            learning_rate=opt.lr,
                            beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1,
                            beta2=opt.b2)


def g_forward(_imgs_A, _labels_A, _valid):
    """GeneratorUNet forward function"""
    # Sample noise as generator input
    z = ops.randn((batch_size, opt.latent_dim))

    # Generate a batch of images
    _fake_B = generator(_imgs_A, z)

    # Perform task on translated source image
    _label_pred = classifier(_fake_B)

    # Calculate the task loss
    task_loss_ = (task_loss(_label_pred, _labels_A) + task_loss(classifier(_imgs_A), _labels_A)) / 2

    # Loss measures generator's ability to fool the discriminator
    _g_loss = lambda_adv * adversarial_loss(discriminator(_fake_B), _valid) + lambda_task * task_loss_

    return _g_loss, _fake_B, _label_pred


def d_forward(_imgs_B, _fake_B, _valid, _fake):
    """Discriminator forward function"""
    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(_imgs_B), _valid)
    fake_loss = adversarial_loss(discriminator(_fake_B), _fake)
    _d_loss = (real_loss + fake_loss) / 2

    return _d_loss


grad_g = ops.value_and_grad(g_forward, None, optimizer_G.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)

generator.set_train()
discriminator.set_train()
classifier.set_train()

# ----------
#  Training
# ----------

# Keeps 100 accuracy measurements
task_performance = []
target_performance = []

for epoch in range(opt.n_epochs):
    for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(dataset1.create_tuple_iterator(),
                                                                     dataset2.create_tuple_iterator())):
        batch_size = imgs_A.shape[0]

        # Adversarial ground truths
        valid = ops.stop_gradient(ops.ones((batch_size, *patch)))
        fake = ops.stop_gradient(ops.zeros((batch_size, *patch)))

        labels_A = ops.Cast()(labels_A, mstype.int32)
        labels_B = ops.Cast()(labels_B, mstype.int32)

        # Configure input
        imgs_A = imgs_A.broadcast_to((batch_size, 3, opt.img_size, opt.img_size))

        # ------------------
        #  Train Generators
        # ------------------

        (g_loss, fake_B, label_pred), g_grads = grad_g(imgs_A, labels_A, valid)
        optimizer_G(g_grads)

        # ----------------------
        #  Train Discriminators
        # ----------------------

        (d_loss), d_grads = grad_d(imgs_B, ops.stop_gradient(fake_B), valid, fake)
        optimizer_D(d_grads)

        # ---------------------------------------
        #  Evaluate Performance on target domain
        # ---------------------------------------

        # Evaluate performance on translated Domain A
        acc = np.mean(np.argmax(label_pred.asnumpy(), axis=1) == labels_A.asnumpy())
        task_performance.append(acc)
        if len(task_performance) > 100:
            task_performance.pop(0)

        # Evaluate performance on Domain B
        pred_B = classifier(imgs_B)
        target_acc = np.mean(np.argmax(pred_B.asnumpy(), axis=1) == labels_B.asnumpy())
        target_performance.append(target_acc)
        if len(target_performance) > 100:
            target_performance.pop(0)

        # --------------
        # Log Progress
        # --------------

        print(
            f'[Epoch {epoch}/{opt.n_epochs}] '
            f'[Batch {i}/{dataset1.get_dataset_size()}] '
            f'[D loss: {d_loss.asnumpy().item():.4f}] '
            f'[G loss: {g_loss.asnumpy().item():.4f}] '
            f'[CLF acc: {100 * acc:3f}% ({100 * np.mean(task_performance):3}%), '
            f'target_acc: {100 * target_acc:3}% ({100 * np.mean(target_performance):3}%)]'
        )

        batches_done = epoch * dataset1.get_dataset_size() + i
        if batches_done % opt.sample_interval == 0:
            sample = ops.cat((imgs_A[:5], fake_B[:5], imgs_B[:5]), -2)
            to_image(sample, os.path.join("images", F'{batches_done}.png'))
