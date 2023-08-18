import torch as T
import torch.nn.functional as F

from src.models.MobileViT.conv_block import ConvBlock


class FireModule(T.nn.Module):
    def __init__(
        self,
        in_channels,
        s1x1,
        e1x1,
        e3x3
    ):
        super(FireModule, self).__init__()

        self.squeeze = ConvBlock(
            in_channels = in_channels,
            out_channels = s1x1,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            is_norm = False,
            is_activation = True
        )

        self.expand = T.nn.ModuleDict({
            "1x1" : ConvBlock(
                in_channels = s1x1,
                out_channels = e1x1,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                is_norm = False,
                is_activation = False
            ),

            "3x3" : ConvBlock(
                in_channels = s1x1,
                out_channels = e3x3,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                is_norm = False,
                is_activation = False
            )
        })


    def forward(self, x):
        x = self.squeeze(x)

        z1 = self.expand["1x1"](x)
        z3 = self.expand["3x3"](x)
        z = T.cat([z1, z3], 1)

        z = F.relu(z)

        return z