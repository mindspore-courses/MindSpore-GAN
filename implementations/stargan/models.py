"""The Models of StarGAN"""

import mindspore.common.initializer as init
from mindspore import nn, ops


##############################
#           RESNET
##############################

class ResidualBlock(nn.Cell):
    """Residual block"""
    def __init__(self, in_features):
        super().__init__(ResidualBlock)

        conv_block = [
            nn.Conv2d(in_features, in_features, 3, 1, 'pad', 1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.InstanceNorm2d(in_features, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, 3, 1, 'pad', 1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.InstanceNorm2d(in_features, affine=True),
        ]

        self.conv_block = nn.SequentialCell(*conv_block)

    def construct(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Cell):
    """Generator Network"""
    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=40):
        super().__init__(GeneratorResNet)
        channels, _, _ = img_shape

        # Initial convolution block
        model = [
            nn.Conv2d(channels + c_dim, 64, 7, 1, 'pad', 3, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU()
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 'pad', 1, has_bias=False,
                          weight_init=init.Normal(0.02, 0.0)),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU()
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.Conv2dTranspose(curr_dim, curr_dim // 2, 4, 2, 'pad', 1, has_bias=False,
                                   weight_init=init.Normal(0.02, 0.0)),
                nn.InstanceNorm2d(curr_dim // 2, affine=True),
                nn.ReLU()
            ]
            curr_dim = curr_dim // 2

        # Output layer
        model += [
            nn.Conv2d(curr_dim, channels, 7, 1, 'pad', 3,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        ]

        self.model = nn.SequentialCell(*model)

    def construct(self, x, c):
        c = c.view(c.shape[0], c.shape[1], 1, 1)
        c = c.tile((1, 1, x.shape[2], x.shape[3]))
        x = ops.cat((x, c), 1)
        return self.model(x)


##############################
#        Discriminator
##############################

class Discriminator(nn.Cell):
    """Discriminator Network"""
    def __init__(self, img_shape=(3, 128, 128), c_dim=40, n_strided=6):
        super().__init__(Discriminator)
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_filters, out_filters, 4, 2, 'pad', 1),
                nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.layers = nn.SequentialCell(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(curr_dim, 1, 3, pad_mode='pad', padding=1, has_bias=False,
                              weight_init=init.Normal(0.02, 0.0))
        # Output 2: Class prediction
        kernel_size = img_size // 2 ** n_strided
        self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, has_bias=False,
                              weight_init=init.Normal(0.02, 0.0),
                              pad_mode='valid')

    def construct(self, img):
        feature_repr = self.layers(img)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return out_adv, out_cls.view(out_cls.shape[0], -1)
