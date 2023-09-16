"""CycleGAN Networks"""

from mindspore import nn
import mindspore.common.initializer as init


##############################
#           RESNET
##############################

class ResidualBlock(nn.Cell):
    def __init__(self, in_features):
        super().__init__(ResidualBlock)

        self.block = nn.SequentialCell(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, has_bias=False,
                      pad_mode='valid',
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(in_features, affine=False),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, has_bias=False,
                      pad_mode='valid',
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(in_features, affine=False)
        )

    def construct(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Cell):
    def __init__(self, input_shape, num_residual_blocks):
        super().__init__(GeneratorResNet)

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7, has_bias=False,
                      pad_mode='valid',
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(out_features, affine=False),
            nn.ReLU()
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, has_bias=False,
                          pad_mode='pad', padding=1,
                          weight_init=init.Normal(0.02, 0.0)),
                nn.BatchNorm2d(out_features, affine=False),
                nn.ReLU()
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
                nn.Conv2d(in_features, out_features, 3, has_bias=False,
                          stride=1, pad_mode='pad', padding=1,
                          weight_init=init.Normal(0.02, 0.0)),
                nn.BatchNorm2d(out_features, affine=False),
                nn.ReLU()
            ]
            in_features = out_features

        # Output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7, has_bias=False,
                      pad_mode='valid',
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        ]

        self.model = nn.SequentialCell(*model)

    def construct(self, x):
        return self.model(x)

    ##############################
    #        Discriminator
    ##############################


class Discriminator(nn.Cell):
    def __init__(self, input_shape):
        super().__init__(Discriminator)

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, has_bias=False,
                                stride=2, pad_mode='pad', padding=1,
                                weight_init=init.Normal(0.02, 0.0))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters, affine=False))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.SequentialCell(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, has_bias=False,
                      pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0))
        )

    def construct(self, img):
        return self.model(img)
