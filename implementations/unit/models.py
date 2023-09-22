"""UNIT Networks"""

import mindspore.common.dtype as mstype
import mindspore.common.initializer as init
from mindspore import nn, ops, Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class LambdaLR(LearningRateSchedule):
    """Learning Rate Schedule"""

    def __init__(self, lr, n_epochs, offset, decay_start_epoch, step_per_epoch):
        super().__init__()
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.lr = lr
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        self.cast = ops.Cast()
        self.step_per_epoch = step_per_epoch

    def construct(self, global_step):
        epoch = self.cast(global_step, mstype.float32) // self.step_per_epoch
        return self.lr * (1.0 - max(Tensor(0.0), epoch + self.offset - self.decay_start_epoch) / (
                self.n_epochs - self.decay_start_epoch))


##############################
#           RESNET
##############################

class ResidualBlock(nn.Cell):
    """Residual block"""

    def __init__(self, features):
        super().__init__(ResidualBlock)

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3, pad_mode='pad',
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(features, affine=False),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3, pad_mode='pad',
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(features, affine=False),
        ]

        self.conv_block = nn.SequentialCell(*conv_block)

    def construct(self, x):
        return x + self.conv_block(x)


class Encoder(nn.Cell):
    """"Encoder"""

    def __init__(self, in_channels=3, dim=64, n_downsample=2, shared_block=None):
        super().__init__(Encoder)

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7, pad_mode='pad',
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(64, affine=False),
            nn.LeakyReLU(0.2),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2,
                          pad_mode='pad', padding=1,
                          weight_init=init.Normal(0.02, 0.0)),
                nn.BatchNorm2d(dim * 2, affine=False),
                nn.ReLU()
            ]
            dim *= 2

        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.SequentialCell(*layers)
        self.shared_block = shared_block

    def reparameterization(self, _mu):
        """Reparameterization func"""
        _z = ops.randn(_mu.shape, dtype=mstype.float32)
        return _z + _mu

    def construct(self, x):
        x = self.model_blocks(x)
        mu = self.shared_block(x)
        z = self.reparameterization(mu)
        return mu, z


class Generator(nn.Cell):
    """Generator Network"""

    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None):
        super().__init__(Generator)

        self.shared_block = shared_block

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.Conv2dTranspose(dim, dim // 2, 4, stride=2,
                                   pad_mode='pad', padding=1,
                                   weight_init=init.Normal(0.02, 0.0)),
                nn.BatchNorm2d(dim // 2, affine=False),
                nn.LeakyReLU(0.2)
            ]
            dim = dim // 2

        # Output layer
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, out_channels, 7,
                      pad_mode="pad", weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        ]

        self.model_blocks = nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.shared_block(x)
        x = self.model_blocks(x)
        return x


##############################
#        Discriminator
##############################


class Discriminator(nn.Cell):
    """Discriminator"""

    def __init__(self, input_shape):
        super().__init__(Discriminator)
        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_filters, out_filters, 4, stride=2,
                          pad_mode='pad', padding=1,
                          weight_init=init.Normal(0.02, 0.0))
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters, affine=False))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.SequentialCell(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 3,
                      pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0))
        )

    def construct(self, img):
        return self.model(img)
