"""The Models of Pix2Pix"""

import mindspore.common.initializer as init
from mindspore import nn
from mindspore import ops


##############################
#           U-NET
##############################

class UNetDown(nn.Cell):
    """U-Net Down"""

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__(UNetDown)
        layers = [
            nn.Conv2d(in_size, out_size, 4, 2,
                      'pad', 1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0))
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=False))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(p=dropout))

        self.model = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.model(x)


class UNetUp(nn.Cell):
    """U-Net Up"""

    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__(UNetUp)
        layers = [
            nn.Conv2dTranspose(in_size, out_size, 4, 2,
                               'pad', 1, has_bias=False,
                               weight_init=init.Normal(0.02, 0.0)),
            nn.InstanceNorm2d(out_size, affine=False),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(p=dropout))

        self.model = nn.SequentialCell(*layers)

    def construct(self, x, skip_input):
        x = self.model(x)
        x = ops.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Cell):
    """U-Net Generator"""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(GeneratorUNet)

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.SequentialCell(
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        )

    def construct(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################

class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self, in_channels=3):
        super().__init__(Discriminator)

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4,
                                stride=2, pad_mode='pad', padding=1,
                                weight_init=init.Normal(0.02, 0.0))]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters, affine=False))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.SequentialCell(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4,
                      pad_mode='pad', padding=1, has_bias=False)
        )

    def construct(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = ops.cat((img_A, img_B), 1)
        return self.model(img_input)
