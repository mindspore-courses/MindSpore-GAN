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
            layers.append(nn.BatchNorm2d(out_size, 0.8,
                                         gamma_init=init.Normal(0.02, 1.0),
                                         beta_init=init.Constant(0.0), affine=False))
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
            nn.BatchNorm2d(out_size, 0.8,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0), affine=False),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(p=dropout))

        self.model = nn.SequentialCell(*layers)

    def construct(self, x, skip_input):
        x = self.model(x)
        x = ops.cat((x, skip_input), 1)

        return x


class Generator(nn.Cell):
    """U-Net Generator"""

    def __init__(self, input_shape):
        super().__init__(Generator)
        channels, _, _ = input_shape
        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128 + channels, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256 + channels, 64)

        final = [
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, channels, 3, 1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        ]
        self.final = nn.SequentialCell(*final)

    def construct(self, x, x_lr):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d2 = ops.cat((d2, x_lr), 1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)


##############################
#        Discriminator
##############################

class Discriminator(nn.Cell):
    """Discriminator Network"""

    def __init__(self, input_shape):
        super().__init__(Discriminator)

        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        patch_h, patch_w = int(height / 2 ** 3), int(width / 2 ** 3)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters,affine=False))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)))

        self.model = nn.SequentialCell(*layers)

    def construct(self, img):
        return self.model(img)
