"""The Models of ESRGAN"""

from mindcv.models import vgg19
from mindspore import nn
from mindspore import ops


class FeatureExtractor(nn.Cell):
    """VGG Feature Extractor"""

    def __init__(self):
        super().__init__(FeatureExtractor)
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.SequentialCell(
            *list(vgg19_model.features.cell_list[:35])
        )

    def construct(self, img):
        return self.vgg19_54(img)


class DenseResidualBlock(nn.Cell):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super().__init__(DenseResidualBlock)
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1,
                                'pad', 1, has_bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.SequentialCell(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def construct(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = ops.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Cell):
    """Residual In Residual Block"""
    def __init__(self, filters, res_scale=0.2):
        super().__init__(ResidualInResidualDenseBlock)
        self.res_scale = res_scale
        self.dense_blocks = nn.SequentialCell(
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
            DenseResidualBlock(filters)
        )

    def construct(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Cell):
    """ResNet Generator"""

    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super().__init__(GeneratorRRDB)

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, 3, 1,
                               'pad', 1)

        # Residual blocks
        self.res_blocks = nn.SequentialCell(
            *[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)]
        )

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, 3, 1,
                               'pad', 1)

        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, 3, 1,
                          'pad', 1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.SequentialCell(*upsample_layers)

        # Final output layer
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(filters, filters, 3, 1, 'pad', 1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, 3, 1, 'pad', 1),
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
            layers.extend(discriminator_block(in_filters, out_filters, first_block=i == 0))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3,
                                1, 'pad', 1))

        self.model = nn.SequentialCell(*layers)

    def construct(self, img):
        return self.model(img)
