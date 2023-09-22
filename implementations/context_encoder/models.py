"""The Models of Context Encoder"""

import mindspore.common.initializer as init
from mindspore import nn


class Generator(nn.Cell):
    """Generator Network"""

    def __init__(self, channels=3):
        super().__init__(Generator)

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 'pad', 1,
                                weight_init=init.Normal(0.02, 0.0))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8,
                                             gamma_init=init.Normal(0.02, 1.0),
                                             beta_init=init.Constant(0.0), affine=False))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2dTranspose(in_feat, out_feat, 4, 2, 'pad', 1,
                                         weight_init=init.Normal(0.02, 0.0))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8,
                                             gamma_init=init.Normal(0.02, 1.0),
                                             beta_init=init.Constant(0.0), affine=False))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.SequentialCell(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            nn.Conv2d(512, 4000, 1,
                      weight_init=init.Normal(0.02, 0.0)),
            *upsample(4000, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            nn.Conv2d(64, channels, 3, 1, 'pad', 1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        )

    def construct(self, x):
        return self.model(x)


class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self, channels=3):
        super().__init__(Discriminator)

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 'pad', 1,
                                weight_init=init.Normal(0.02, 0.0))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=False))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 'pad', 1,
                                weight_init=init.Normal(0.02, 0.0)))

        self.model = nn.SequentialCell(*layers)

    def construct(self, img):
        return self.model(img)
