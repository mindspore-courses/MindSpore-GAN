"""ClusterGAN Model"""

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
import mindspore.numpy as mnp
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

parser = argparse.ArgumentParser(description="ClusterGAN Training Script")
parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
parser.add_argument("-i", "--img_size", dest="img_size", type=int, default=28, help="Size of image dimension")
parser.add_argument("-d", "--latent_dim", dest="latent_dim", default=30, type=int, help="Dimension of latent space")
parser.add_argument("-l", "--lr", dest="learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("-c", "--n_critic", dest="n_critic", type=int, default=5,
                    help="Number of training steps for discriminator per iter")
parser.add_argument("-w", "--wass_flag", dest="wass_flag", action='store_true', help="Flag for Wasserstein metric")
args = parser.parse_args()


def sample_z(shape=64, _latent_dim=10, _n_c=10, fix_class=-1):
    """Sample a random latent space vector"""
    assert (fix_class == -1 or (0 <= fix_class < _n_c)), f'Requested class {fix_class} outside bounds.'

    # Sample noise as generator input, zn
    zn = 0.75 * ops.stop_gradient(ops.randn((shape, _latent_dim), dtype=mstype.float32))

    ######### zc, zc_idx variables with grads, and zc to one-hot vector
    # Pure one-hot vector generation
    zc_FT = ops.zeros((shape, _n_c), dtype=mstype.float32)
    zc_idx = Tensor(mnp.empty(shape, dtype=mstype.int64))

    if fix_class == -1:
        zc_idx = ops.uniform(zc_idx.shape, minval=Tensor(0, dtype=mstype.int32),
                             maxval=Tensor(_n_c, dtype=mstype.int32), dtype=mstype.int32)
        zc_FT = zc_FT.scatter(1, zc_idx.unsqueeze(1), ops.ones((zc_idx.shape[0], 1), dtype=mstype.float32))
    else:
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

        zc_FT = ops.stop_gradient(zc_FT)

    zc = zc_FT

    # Return components of latent space variable
    return zn, zc, zc_idx


def calc_gradient_penalty(D, real_data, generated_data):
    """Calculate gradient penalty"""
    # GP strength
    LAMBDA = 10

    b_size = real_data.shape[0]

    # Calculate interpolation
    alpha = ops.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)

    interpolated = alpha * real_data + (1 - alpha) * generated_data

    # Calculate gradients of probabilities with respect to examples
    grad_fn = ops.grad(D)
    gradients = grad_fn(interpolated)

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = ops.sqrt(ops.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


def softmax(x):
    """Softmax function"""
    return ops.softmax(x, axis=1)


class Reshape(nn.Cell):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=None):
        super().__init__(Reshape)
        if shape is None:
            shape = []
        self.shape = shape

    def construct(self, x):
        return x.view(x.shape[0], *self.shape)


class GeneratorCNN(nn.Cell):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """

    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, _latent_dim, _n_c, _x_shape, verbose=False):
        super().__init__(GeneratorCNN)

        self.name = 'generator'
        self.latent_dim = _latent_dim
        self.n_c = _n_c
        self.x_shape = _x_shape
        self.ishape = (128, 7, 7)
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose

        self.model = nn.SequentialCell(
            # Fully connected layers
            nn.Dense(self.latent_dim + self.n_c, 1024,
                     weight_init=init.Normal(0.02, 0),
                     bias_init=init.Constant(0)),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dense(1024, self.iels,
                     weight_init=init.Normal(0.02, 0),
                     bias_init=init.Constant(0)),
            nn.BatchNorm1d(self.iels),
            nn.LeakyReLU(0.2),

            # Reshape to 128 x (7x7)
            Reshape(self.ishape),

            # Upconvolution layers
            nn.Conv2dTranspose(128, 64, 4, stride=2,
                               pad_mode='pad', padding=1, has_bias=True,
                               weight_init=init.Normal(0.02, 0),
                               bias_init=init.Constant(0)),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.2),

            nn.Conv2dTranspose(64, 1, 4, stride=2,
                               pad_mode='pad', padding=1, has_bias=True,
                               weight_init=init.Normal(0.02, 0),
                               bias_init=init.Constant(0)),
            nn.Sigmoid()
        )

        if self.verbose:
            print(f'Setting up {self.name}...\n')
            print(self.model)

    def construct(self, zn, zc):
        z = ops.cat((zn, zc), 1)
        x_gen = self.model(z)
        # Reshape for output
        x_gen = x_gen.view(x_gen.shape[0], *self.x_shape)
        return x_gen


class EncoderCNN(nn.Cell):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """

    def __init__(self, _latent_dim, _n_c, verbose=False):
        super().__init__(EncoderCNN)

        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = _latent_dim
        self.n_c = _n_c
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose

        self.model = nn.SequentialCell(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2,
                      has_bias=True, pad_mode='valid',
                      weight_init=init.Normal(0.02, 0),
                      bias_init=init.Constant(0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2,
                      has_bias=True, pad_mode='valid',
                      weight_init=init.Normal(0.02, 0),
                      bias_init=init.Constant(0)),
            nn.LeakyReLU(0.2),

            # Flatten
            Reshape(self.lshape),

            # Fully connected layers
            nn.Dense(self.iels, 1024,
                     weight_init=init.Normal(0.02, 0),
                     bias_init=init.Constant(0)),
            nn.LeakyReLU(0.2),
            nn.Dense(1024, _latent_dim + _n_c,
                     weight_init=init.Normal(0.02, 0),
                     bias_init=init.Constant(0))
        )

        if self.verbose:
            print(f'Setting up {self.name}...\n')
            print(self.model)

    def construct(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)
        # Separate continuous and one-hot components
        zn = z[:, 0:self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        # Softmax on zc component
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class DiscriminatorCNN(nn.Cell):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is GeneratorUNet(z)
    Output is a 1-dimensional value
    """

    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, _wass_metric=False, verbose=False):
        super().__init__(DiscriminatorCNN)

        self.name = 'discriminator'
        self.channels = 1
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = _wass_metric
        self.verbose = verbose

        self.model = nn.SequentialCell(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4,
                      stride=2, has_bias=True, pad_mode='pad',
                      weight_init=init.Normal(0.02, 0),
                      bias_init=init.Constant(0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4,
                      stride=2, has_bias=True, pad_mode='pad',
                      weight_init=init.Normal(0.02, 0),
                      bias_init=init.Constant(0)),
            nn.LeakyReLU(0.2),

            # Flatten
            Reshape(self.lshape),

            # Fully connected layers
            nn.Dense(self.iels, 1024,
                     weight_init=init.Normal(0.02, 0),
                     bias_init=init.Constant(0)),
            nn.LeakyReLU(0.2),
            nn.Dense(1024, 1,
                     weight_init=init.Normal(0.02, 0),
                     bias_init=init.Constant(0)),
        )

        # If NOT using Wasserstein metric, final Sigmoid
        if not self.wass:
            self.model = nn.SequentialCell(self.model, nn.Sigmoid())

        if self.verbose:
            print(f'Setting up {self.name}...\n')
            print(self.model)

    def construct(self, img):
        # Get output
        validity = self.model(img)
        return validity


# Training details
n_epochs = args.n_epochs
batch_size = args.batch_size
test_batch_size = 5000
lr = args.learning_rate
b1 = 0.5
b2 = 0.9
decay = 2.5 * 1e-5
n_skip_iter = args.n_critic

# Data dimensions
img_size = args.img_size
channels = 1

# Latent space info
latent_dim = args.latent_dim
n_c = 10
betan = 10
betac = 10

# Wasserstein+GP metric flag
wass_metric = args.wass_flag

x_shape = (channels, img_size, img_size)

# Loss function
bce_loss = nn.BCELoss()
xe_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()

# Initialize generator and discriminator
generator = GeneratorCNN(latent_dim, n_c, x_shape)
encoder = EncoderCNN(latent_dim, n_c)
discriminator = DiscriminatorCNN(_wass_metric=wass_metric)

generator.update_parameters_name("generator")
encoder.update_parameters_name("encoder")
discriminator.update_parameters_name("discriminator")

param_G = list(generator.trainable_params()) + list(encoder.trainable_params())

ge_chain = itertools.chain(generator.trainable_params(),
                           encoder.trainable_params())

optimizer_GE = nn.optim.Adam(ge_chain, learning_rate=lr, beta1=b1, beta2=b2, weight_decay=decay)
optimizer_D = nn.optim.Adam(discriminator.trainable_params(), learning_rate=lr, beta1=b1, beta2=b2)

transform = [
    transforms.ToTensor(),
]

dataset = mindspore.dataset.MnistDataset(
    dataset_dir="../../data/MNIST",
    usage='train',
    shuffle=True
).map(operations=transform, input_columns="image").batch(batch_size, drop_remainder=True)

testdata = mindspore.dataset.MnistDataset(
    dataset_dir="../../data/MNIST",
    usage='test',
    shuffle=True
).map(operations=transform, input_columns="image").batch(batch_size)

test_imgs, test_labels = next(testdata.create_tuple_iterator())


def ge_forward(_real_imgs):
    """GeneratorUNet forward function"""
    # Sample random latent variables
    zn, zc, zc_idx = sample_z(shape=_real_imgs.shape[0], _latent_dim=latent_dim, _n_c=n_c)
    # Generate a batch of images
    _gen_imgs = generator(zn, zc)
    # Discriminator output from real and generated samples
    D_gen = discriminator(_gen_imgs)
    # Encode the generated images
    enc_gen_zn, _, enc_gen_zc_logits = encoder(_gen_imgs)

    # Calculate losses for z_n, z_c
    zn_loss = mse_loss(enc_gen_zn, zn)
    zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)

    # Check requested metric
    if wass_metric:
        # Wasserstein GAN loss
        _ge_loss = ops.mean(D_gen) + betan * zn_loss + betac * zc_loss
    else:
        # Vanilla GAN loss
        _valid = ops.stop_gradient(ops.ones((_gen_imgs.shape[0], 1), dtype=mstype.float32))
        v_loss = bce_loss(D_gen, _valid)
        _ge_loss = v_loss + betan * zn_loss + betac * zc_loss

    return _ge_loss, _gen_imgs, _valid


def d_forward(_real_imgs, _gen_imgs, _valid):
    """Discriminator forward function"""
    # Discriminator output from real and generated samples
    _D_gen = discriminator(_gen_imgs)
    _D_real = discriminator(real_imgs)
    # Measure discriminator's ability to classify real from generated samples
    if wass_metric:
        # Gradient penalty term
        grad_penalty = calc_gradient_penalty(discriminator, _real_imgs, _gen_imgs)

        # Wasserstein GAN loss w/gradient penalty
        _d_loss = ops.mean(_D_real) - ops.mean(_D_gen) + grad_penalty

    else:
        # Vanilla GAN loss
        fake = ops.stop_gradient(ops.zeros((_gen_imgs.shape[0], 1), dtype=mstype.float32))
        real_loss = bce_loss(_D_real, _valid)
        fake_loss = bce_loss(_D_gen, fake)
        _d_loss = (real_loss + fake_loss) / 2

    return _d_loss


grad_ge = ops.value_and_grad(ge_forward, None, optimizer_GE.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, optimizer_D.parameters, has_aux=False)

# ----------
#  Training
# ----------
ge_l = []
d_l = []

c_zn = []
c_zc = []
c_i = []

# Training loop
print(f'\nBegin training session with {n_epochs} epochs...\n')
for epoch in range(n_epochs):
    itruth_label = None
    for i, (imgs, label) in enumerate(dataset.create_tuple_iterator()):
        itruth_label = label
        # Ensure generator/encoder are trainable
        generator.set_train()
        discriminator.set_train()

        # Configure input
        real_imgs = imgs

        # ---------------------------
        #  Train GeneratorUNet + Encoder
        # ---------------------------

        # Step for GeneratorUNet & Encoder, n_skip_iter times less than for discriminator

        (ge_loss, gen_imgs, valid), ge_grads = grad_ge(real_imgs)
        if i % n_skip_iter == 0:
            optimizer_GE(ge_grads)
        # ---------------------
        #  Train Discriminator
        # ---------------------

        (d_loss), d_grads = grad_d(real_imgs, gen_imgs, valid)
        optimizer_D(d_grads)

    # Save training losses
    d_l.append(d_loss.asnumpy().item())
    ge_l.append(ge_loss.asnumpy().item())

    generator.set_train(False)
    discriminator.set_train(False)

    # Set number of examples for cycle calcs
    n_sqrt_samp = 5
    n_samp = n_sqrt_samp * n_sqrt_samp

    ## Cycle through test real -> enc -> gen
    t_imgs, t_label = test_imgs, test_labels
    # Encode sample real instances
    e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)
    # Generate sample instances from encoding
    teg_imgs = generator(e_tzn, e_tzc)
    # Calculate cycle reconstruction loss
    img_mse_loss = mse_loss(t_imgs, teg_imgs)
    # Save img reco cycle loss
    c_i.append(img_mse_loss.asnumpy().item())

    ## Cycle through randomly sampled encoding -> generator -> encoder
    zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp, _latent_dim=latent_dim, _n_c=n_c)
    # Generate sample instances
    gen_imgs_samp = generator(zn_samp, zc_samp)

    # Encode sample instances
    zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp)

    # Calculate cycle latent losses
    lat_mse_loss = mse_loss(zn_e, zn_samp)
    lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)

    # Save latent space cycle losses
    c_zn.append(lat_mse_loss.asnumpy().item())
    c_zc.append(lat_xe_loss.asnumpy().item())

    # Save cycled and generated examples!
    r_imgs, i_label = real_imgs[:n_samp], itruth_label[:n_samp]
    e_zn, e_zc, e_zc_logits = encoder(r_imgs)
    reg_imgs = generator(e_zn, e_zc)

    to_image(reg_imgs, f'images/cycle_reg_{epoch:06}.png')
    to_image(gen_imgs_samp, f'images/gen_{epoch:06}.png')

    ## Generate samples for specified classes
    stack_imgs = []
    for idx in range(n_c):
        # Sample specific class
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_c, _latent_dim=latent_dim, _n_c=n_c, fix_class=idx)

        # Generate sample instances
        gen_imgs_samp = generator(zn_samp, zc_samp)

        if len(stack_imgs) == 0:
            stack_imgs = gen_imgs_samp
        else:
            stack_imgs = ops.cat((stack_imgs, gen_imgs_samp), 0)

    # Save class-specified generated examples!
    to_image(stack_imgs, f'images/gen_classes_{epoch:06}.png')

    print(f'[Epoch {epoch}/{n_epochs}] '
          f'Model Losses: [D: {d_loss.asnumpy().item():.4f}] '
          f'[GE: {ge_loss.asnumpy().item():.4f}]'
          )

    print(f'Cycle Losses: [x: {img_mse_loss.asnumpy().item():.4f}] '
          f'[z_n: {lat_mse_loss.asnumpy().item():.4f}] '
          f'[z_c: {lat_xe_loss.asnumpy().item():.4f}]'
          )
