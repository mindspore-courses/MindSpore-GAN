import argparse
import gzip
import os
import shutil
import urllib
from urllib import request

import mindspore
import numpy as np
from matplotlib import pyplot as plt
from mindspore import Tensor, ops
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import transforms

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
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

image_path = "./images"


# Save the generated test image.
def save_imgs(gen_imgs1, idx):
    for i3 in range(gen_imgs1.shape[0]):
        plt.subplot(5, 5, i3 + 1)
        plt.imshow(gen_imgs1[i3, 0, :, :] / 2 + 0.5, cmap="gray")
        plt.axis("off")
    plt.savefig(image_path + "/test_{}.png".format(idx))


class Generator(nn.Cell):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Dense(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
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

    def construct(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.SequentialCell(
            nn.Dense(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dense(256, 1)
        )

    def construct(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


generator = Generator()
discriminator = Discriminator()
generator.update_parameters_name('generator')
discriminator.update_parameters_name('discriminator')
generator.set_train()
discriminator.set_train()

# Loss weight for gradient penalty
lambda_gp = 10

G_Optim = nn.optim.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
D_Optim = nn.optim.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)

G_Optim.update_parameters_name('G_Optim')
D_Optim.update_parameters_name('D_Optim')


def compute_gradient_penalty(real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = ops.randn((real_samples.shape[0], 1, 1, 1))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))

    grad_fn = ops.grad(discriminator)
    # Get gradient w.r.t. interpolates
    gradients = grad_fn(interpolates)

    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((mindspore.numpy.norm(gradients, 2, axis=1) - 1) ** 2).mean()
    return gradient_penalty


def g_forward(z):
    fake_imgs = generator(z)
    # Loss measures generator's ability to fool the discriminator
    # Train on fake images
    fake_validity = discriminator(fake_imgs)
    g_loss = -ops.mean(fake_validity)
    return g_loss, fake_imgs


# 判别器正向传播
def d_forward(real_imgs):
    z = ops.StandardNormal()((imgs.shape[0], opt.latent_dim))
    fake_imgs = generator(z)
    # Real images
    real_validity = discriminator(real_imgs)
    # Fake images
    fake_validity = discriminator(fake_imgs)
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(real_imgs, fake_imgs)
    # Adversarial loss
    d_loss = -ops.mean(real_validity) + ops.mean(fake_validity) + lambda_gp * gradient_penalty

    return d_loss, z


transform = [
    transforms.Rescale(1.0 / 255.0, 0),
    transforms.Resize(opt.img_size),
    # transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.HWC2CHW()
]

dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='train',
    shuffle=True
).map(operations=transform, input_columns="image").batch(opt.batch_size)

grad_g = ops.value_and_grad(g_forward, None, G_Optim.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, D_Optim.parameters, has_aux=True)


batches_done = 0

generator.update_parameters_name('generator')
discriminator.update_parameters_name('discriminator')
discriminator.set_train()
generator.set_train()

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        real_imgs = Tensor(imgs, dtype=mstype.float32)

        (loss_D, z), D_grads = grad_d(real_imgs)
        D_Optim(D_grads)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            (loss_G, gen_imgs), G_grads = grad_g(z)
            G_Optim(G_grads)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, dataset.get_dataset_size(), loss_D.asnumpy().item(), loss_G.asnumpy().item())
            )

            if batches_done % opt.sample_interval == 0:
                save_imgs(gen_imgs[:25].asnumpy(), batches_done)
                # to_image(gen_imgs[:25], os.path.join("images", F'{batches_done}.png'))
            batches_done += opt.n_critic
