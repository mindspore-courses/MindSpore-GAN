"""MUNIT Models"""

from mindspore import nn, ops, Parameter, Tensor
import mindspore.common.initializer as init
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
import mindspore.common.dtype as mstype


class LambdaLR(LearningRateSchedule):
    """Learning rate decay"""

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


#################################
#           Encoder
#################################


class Encoder(nn.Cell):
    """Encoder Network"""

    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2, style_dim=8):
        super().__init__(Encoder)
        self.content_encoder = ContentEncoder(in_channels, dim, n_residual, n_downsample)
        self.style_encoder = StyleEncoder(in_channels, dim, n_downsample, style_dim)

    def construct(self, x):
        content_code = self.content_encoder(x)
        style_code = self.style_encoder(x)
        return content_code, style_code


#################################
#            Decoder
#################################


class Decoder(nn.Cell):
    """Decoder Network"""

    def __init__(self, out_channels=3, dim=64, n_residual=3, n_upsample=2, style_dim=8):
        super().__init__(Decoder)

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="adain")]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2.0, recompute_scale_factor=True),
                nn.Conv2d(dim, dim // 2, 5, stride=1, pad_mode='pad', padding=2,
                          weight_init=init.Normal(0.02, 0.0)),
                LayerNorm(dim // 2),
                nn.ReLU()
            ]
            dim = dim // 2

        # Output layer
        layers += [
            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn.Conv2d(dim, out_channels, 7, pad_mode='valid',
                      weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        ]

        self.model = nn.SequentialCell(*layers)

        # Initiate mlp (predicts AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for _, cell in self.cells_and_names():
            if cell.cls_name == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * cell.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for _, cell in self.cells_and_names():
            if cell.cls_name == "AdaptiveInstanceNorm2d":
                # Extract mean and std predictions
                mean = adain_params[:, : cell.num_features]
                std = adain_params[:, cell.num_features: 2 * cell.num_features]
                # Update bias and weight
                cell.bias = mean.view(-1)
                cell.weight = std.view(-1)
                # Move pointer
                if adain_params.shape[1] > 2 * cell.num_features:
                    adain_params = adain_params[:, 2 * cell.num_features:]

    def construct(self, content_code, style_code):
        # Update AdaIN parameters by MLP prediction based off style code
        self.assign_adain_params(self.mlp(style_code))
        img = self.model(content_code)
        return img


#################################
#        Content Encoder
#################################

class ContentEncoder(nn.Cell):
    """Content Encoder"""

    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super().__init__(ContentEncoder)

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7, pad_mode='valid',
                      weight_init=init.Normal(0.02, 0.0)),
            nn.InstanceNorm2d(dim, affine=False),
            nn.ReLU()
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, pad_mode='pad', padding=1,
                          weight_init=init.Normal(0.02, 0.0)),
                nn.InstanceNorm2d(dim * 2, affine=False),
                nn.ReLU()
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="in")]

        self.model = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.model(x)


#################################
#        Style Encoder
#################################


class StyleEncoder(nn.Cell):
    """Style Encoder"""
    def __init__(self, in_channels=3, dim=64, n_downsample=2, style_dim=8):
        super().__init__(StyleEncoder)

        # Initial conv block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7, pad_mode='valid',
                      weight_init=init.Normal(0.2, 0.0)),
            nn.ReLU()
        ]

        # Downsampling
        for _ in range(2):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, pad_mode='pad', padding=1,
                          weight_init=init.Normal(0.2, 0.0)),
                nn.ReLU()
            ]
            dim *= 2

        # Downsampling with constant depth
        for _ in range(n_downsample - 2):
            layers += [
                nn.Conv2d(dim, dim, 4, stride=2, padding=1, pad_mode='pad',
                          weight_init=init.Normal(0.2, 0.0)),
                nn.ReLU()
            ]

        # Average pool and output layer
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, style_dim, 1, 1,
                      'pad', 0,
                      weight_init=init.Normal(0.2, 0.0))]

        self.model = nn.SequentialCell(*layers)

    def construct(self, x):
        # x = nn.ReflectionPad2d(3)(x)
        # x = nn.Conv2d(3, 64, 7, pad_mode='valid',
        #           weight_init=init.Normal(0.2, 0.0))(x)
        # x = nn.ReLU()(x)
        # x = nn.Conv2d(128, 128, 4, stride=2, padding=1, pad_mode='pad',
        #           weight_init=init.Normal(0.2, 0.0))(x)
        return self.model(x)


######################################
#   MLP (predicts AdaIn parameters)
######################################


class MLP(nn.Cell):
    """MLP model"""

    def __init__(self, input_dim, output_dim, dim=256, n_blk=3):
        super().__init__(MLP)
        layers = [nn.Dense(input_dim, dim), nn.ReLU()]
        for _ in range(n_blk - 2):
            layers += [nn.Dense(dim, dim), nn.ReLU()]
        layers += [nn.Dense(dim, output_dim)]
        self.model = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.model(x.view(x.shape[0], -1))


##############################
#        Discriminator
##############################


class MultiDiscriminator(nn.Cell):
    """Multi-Discriminator"""
    def __init__(self, in_channels=3):
        super().__init__(MultiDiscriminator)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2,
                                pad_mode='pad', padding=1,
                                weight_init=init.Normal(0.2, 0.0))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=False))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        # Extracts three discriminator models
        self.models = nn.CellList()
        for _ in range(3):
            self.models.append(
                nn.SequentialCell(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3,
                              padding=1, pad_mode='pad',
                              weight_init=init.Normal(0.2, 0.0))
                )
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], pad_mode='pad', count_include_pad=False)

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


##############################
#       Custom Blocks
##############################


class ResidualBlock(nn.Cell):
    """Residual blocks"""

    def __init__(self, features, norm="in"):
        super().__init__(ResidualBlock)

        norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d

        self.block = nn.SequentialCell(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(features, features, 3, pad_mode='valid',
                      weight_init=init.Normal(0.2, 0.0)),
            norm_layer(features),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(features, features, 3, pad_mode='valid',
                      weight_init=init.Normal(0.2, 0.0)),
            norm_layer(features)
        )

    def construct(self, x):
        return x + self.block(x)


##############################
#        Custom Layers
##############################


class AdaptiveInstanceNorm2d(nn.Cell):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__(AdaptiveInstanceNorm2d)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = init.initializer(init.Normal(0.2, 0.0), (256,))
        self.bias = init.initializer(init.Normal(0.02, 0.0), (256,))
        # just dummy buffers, not used
        self.running_var = Parameter(ops.ones(num_features), requires_grad=False)
        self.running_mean = Parameter(ops.zeros(num_features), requires_grad=False)

    def construct(self, x):
        assert (
                self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.shape
        running_mean = self.running_mean.tile((b,))
        running_var = self.running_var.tile((b,))

        # Apply instance norm
        x_reshaped = x.view(1, b * c, h, w)

        out = ops.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Cell):
    """LayerNorm"""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__(LayerNorm)
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = (Parameter(ops.rand(num_features)))
            self.beta = (Parameter(ops.zeros(num_features)))

    def construct(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.shape[0], -1).mean(1).view(*shape)
        std = x.view(x.shape[0], -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
