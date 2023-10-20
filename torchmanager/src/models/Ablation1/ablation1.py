import torch as T

from src.models.SqueezeViT_3.fire_block import FireBlock
from src.models.SqueezeViT_3.svit_block import SqueezeViTBlock
from src.models.MobileViT.conv_block import ConvBlock


class Ablation1(T.nn.Module):
    def __init__(
        self,
        image_size, # c, h, w
        squeeze_layers,
        channels,

        embed_dim,
        num_blocks,
        num_heads,
        mlp_scale,
        patch_size,
        num_classes,

        qkv_bias = True,
        attn_p = 0.,
        proj_p = 0.
    ):
        super(Ablation1, self).__init__()

        in_channels, ih, iw = image_size


        self.pooler_conv = ConvBlock(
            in_channels = in_channels,
            out_channels = 3,
            kernel_size = 3,
            stride = 2,
            padding = 0,
            is_norm = False,
            is_activation = True
        )
        self.counter_pool = T.nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            padding = 1
        )

        self.fire1 = FireBlock(
            num_layers = 1,
            channels = [3, 128], #channels[0]
            is_pool = False
        )

        # self.fire2 = FireBlock(
        #     num_layers = squeeze_layers[0],
        #     channels = channels[:squeeze_layers[0] + 1],
        #     is_pool = True
        # )

        self.svit1 = SqueezeViTBlock(
            in_channels = channels[squeeze_layers[0]],
            out_channels = embed_dim,
            num_blocks = num_blocks,
            num_heads = num_heads,
            mlp_scale = mlp_scale,
            patch_size = patch_size,
            qkv_bias = qkv_bias,
            attn_p = attn_p,
            proj_p = proj_p
        )

        self.fire3 = FireBlock(
            num_layers = squeeze_layers[1]-1,
            channels = channels[squeeze_layers[0] :],
            is_pool = True
        )

        self.conv = ConvBlock(
            in_channels = channels[-1],
            out_channels = channels[-1],
            kernel_size = 1,
            stride = 1,
            padding = 0,
            is_norm = True,
            is_activation = True
        )
        # self.fire4 = FireBlock(
        #     num_layers = 1,
        #     channels = [channels[-1], channels[-1]],
        #     is_pool = False
        # )


        self.classifier = T.nn.Sequential(
            T.nn.AdaptiveAvgPool2d((1, 1)),
            T.nn.Flatten(),
            T.nn.Linear(channels[-1], num_classes, bias=True),
            T.nn.Sigmoid()
        )


    def forward(self, x):
        x = self.pooler_conv(x)
        x = self.counter_pool(x)
        x = self.fire1(x)

        # x = self.fire2(x)
        x = self.svit1(x)
        x = self.fire3(x)
        x = self.conv(x)
        # x = self.fire4(x)

        x = self.classifier(x)
        return x