import torch as T

from src.models.SqueezeViT_3.fire_module import FireModule


class FireBlock(T.nn.Module):
    def __init__(
        self,
        num_layers,
        channels,
        is_pool
    ):
        super(FireBlock, self).__init__()

        channels_half = [channel//2 for channel in channels]
        layers = [
            FireModule(
                in_channels = channels[i],
                s1x1 = channels_half[i],
                e1x1 = channels_half[i+1],
                e3x3 = channels[i+1] - channels_half[i+1]
            )
            for i in range(num_layers)
        ]
        if is_pool:
            layers.append(
                T.nn.MaxPool2d(
                    kernel_size = 3,
                    stride = 2,
                    padding = 1
                )
            )

        self.squeeze_block = T.nn.Sequential(*layers)


    def forward(self, x):
        x = self.squeeze_block(x)
        return x