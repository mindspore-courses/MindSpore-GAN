"""Networks"""
from mindcv import resnet18
from mindspore import nn, ops
import mindspore.common.initializer as init


##############################
#           U-NET
##############################


class UNetDown(nn.Cell):
    """U-Net Down"""

    def __init__(self, in_size, out_size, normalize=True):
        super().__init__(UNetDown)
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, pad_mode='pad', padding=1, has_bias=False,
                            weight_init=init.Normal(0.02, 0.0))]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8, affine=False,
                                         gamma_init=init.Normal(0.02, 1.0),
                                         beta_init=init.Constant(0.0)))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.model(x)


class UNetUp(nn.Cell):
    """U-Net Up"""

    def __init__(self, in_size, out_size):
        super().__init__(UNetUp)
        self.model = nn.SequentialCell(
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(in_size, out_size, 3, stride=1, pad_mode='pad', padding=1, has_bias=False,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(out_size, 0.8, affine=False,
                           gamma_init=init.Normal(0.02, 1.0),
                           beta_init=init.Constant(0.0)),
            nn.ReLU()
        )

    def construct(self, x, skip_input):
        x = self.model(x)
        x = ops.cat((x, skip_input), 1)
        return x


class Generator(nn.Cell):
    """GeneratorUNet Network"""

    def __init__(self, latent_dim, img_shape):
        super().__init__(Generator)
        channels, self.h, self.w = img_shape

        self.fc = nn.Dense(latent_dim, self.h * self.w)

        self.down1 = UNetDown(channels + 1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.SequentialCell(
            nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
            nn.Conv2d(128, channels, 3, stride=1, pad_mode='pad', padding=1,
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        )

    def construct(self, x, z):
        # Propogate noise through fc layer and reshape to img shape
        z = self.fc(z).view(z.shape[0], 1, self.h, self.w)
        d1 = self.down1(ops.cat((x, z), 1))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)


##############################
#        Encoder
##############################


class Encoder(nn.Cell):
    """Encoder Network"""

    def __init__(self, latent_dim):
        super().__init__(Encoder)
        resnet18_model = resnet18(pretrained=False)
        self.feature_extractor = nn.SequentialCell(*list(resnet18_model.cells())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Dense(256, latent_dim)
        self.fc_logvar = nn.Dense(256, latent_dim)

    def construct(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.shape[0], -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


##############################
#        Discriminator
##############################


class MultiDiscriminator(nn.Cell):
    """Multi-Discriminator"""
    def __init__(self, input_shape):
        super().__init__(MultiDiscriminator)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, pad_mode='pad', padding=1,
                                weight_init=init.Normal(0.02, 0.0))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters, 0.8,
                                             gamma_init=init.Normal(0.02, 1.0),
                                             beta_init=init.Constant(0.0)))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        channels, _, _ = input_shape
        # Extracts discriminator models
        self.models = nn.CellList()
        for _ in range(3):
            self.models.append(
                nn.SequentialCell(
                    *discriminator_block(channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, pad_mode='pad', padding=1,
                              weight_init=init.Normal(0.02, 0.0))
                )
            )

        self.downsample = nn.AvgPool2d(channels, stride=2, pad_mode='pad', padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([ops.mean((out - gt) ** 2) for out in self.construct(x)])
        return loss

    def construct(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs
