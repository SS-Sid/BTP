import torch as T

class ConvBlock(T.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        is_norm = True,
        is_activation = True,
        **kwargs
    ):
        super(ConvBlock, self).__init__()


        layers = []
        # conv2d
        layers.append(
            T.nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                bias = False,
                **kwargs
            )
        )

        # bn2d
        if is_norm:
            layers.append(
                T.nn.BatchNorm2d(out_channels)
            )

        # activation
        if is_activation:
            layers.append(
                T.nn.SiLU()
            )

        self.conv_bn_act = T.nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv_bn_act(x)
        return x