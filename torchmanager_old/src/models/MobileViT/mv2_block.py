import torch as T
from src.models.MobileViT.conv_block import ConvBlock

class MV2Block(T.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expansion_factor
    ):
        super(MV2Block, self).__init__()

        self.is_skip_connection = stride == 1 and in_channels == out_channels

        hidden_channels = int(round(in_channels * expansion_factor))

        layers = []
        if expansion_factor != 1:
            # point-wise
            layers.append(
                ConvBlock(
                    in_channels = in_channels,
                    out_channels = hidden_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = 0,
                    is_norm = True,
                    is_activation = True
                )
            )

        layers.extend([
            # depth-wise
            ConvBlock(
                in_channels = hidden_channels,
                out_channels = hidden_channels,
                kernel_size = 3,
                stride = stride,
                padding = 1,
                groups = hidden_channels,
                is_norm = True,
                is_activation = True
            ),
            # point-wise-linear
            ConvBlock(
                in_channels = hidden_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                is_norm = True,
                is_activation = False
            )
        ])
        self.mv2 = T.nn.Sequential(*layers)


    def forward(self, x):
        y = self.mv2(x)

        if self.is_skip_connection:
            y = x + y

        return y