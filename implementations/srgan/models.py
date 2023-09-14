"""The Models of SRGAN"""

import mindspore.common.initializer as init
from mindspore import nn
from mindspore import ops
from mindcv.models import vgg19


class FeatureExtractor(nn.Cell):
    """VGG Feature Extractor"""

    def __init__(self):
        super().__init__(FeatureExtractor)
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.SequentialCell(
            *list(vgg19_model.features.cell_list[:18])
        )

    def construct(self, img):
        return self.feature_extractor(img)


class ResidualBlock(nn.Cell):
    """Residual Block"""

    def __init__(self, in_features):
        super().__init__(ResidualBlock)
        self.conv_block = nn.SequentialCell(
            nn.Conv2d(in_features, in_features,
                      3, 1, 'pad', 1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features,
                      3, 1, 'pad', 1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def construct(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Cell):
    """ResNet Generator"""

    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super().__init__(GeneratorResNet)

        # First layer
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_channels, 64,
                      9, 1, 'pad', 4),
            nn.PReLU()
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.SequentialCell(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(64, 64,
                      3, 1, 'pad', 1),
            nn.BatchNorm2d(64, 0.8)
        )

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256,
                          3, 1, 'pad', 1),
                nn.BatchNorm2d(256, affine=False),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]
        self.upsampling = nn.SequentialCell(*upsampling)

        # Final output layer
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(64, out_channels,
                      9, 1, 'pad', 4),
            nn.Tanh()
        )

    def construct(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = ops.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self, input_shape):
        super().__init__(Discriminator)

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, 3,
                                    1, 'pad', 1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters, affine=False))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Conv2d(out_filters, out_filters, 3,
                                    2, 'pad', 1))
            layers.append(nn.BatchNorm2d(out_filters, affine=False))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3,
                                1, 'pad', 1))

        self.model = nn.SequentialCell(*layers)

    def construct(self, img):
        return self.model(img)
