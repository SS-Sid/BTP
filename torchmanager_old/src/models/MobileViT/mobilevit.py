import torch as T
from src.models.MobileViT.conv_block import ConvBlock
from src.models.MobileViT.mv2_block import MV2Block
from src.models.MobileViT.mobilevit_block import MobileViTBlock


class MobileViT(T.nn.Module):
    def __init__(
        self,
        image_size, # c, h, w
        channels,
        expansion_factor,
        dims,
        kernel_size,
        patch_size,
        num_classes,
        qkv_bias = True,
        attn_p = 0.,
        proj_p = 0.
    ):
        super(MobileViT, self).__init__()

        in_channels, ih, iw = image_size

        self.conv1 = ConvBlock(
            in_channels = in_channels,
            out_channels = channels[0],
            kernel_size = 3,
            stride = 2,
            padding = 1,
            is_norm = True,
            is_activation = True
        )

        self.mv2_1 = MV2Block(
            in_channels = channels[0],
            out_channels = channels[1],
            stride = 1,
            expansion_factor = expansion_factor
        )
        self.mv2_2 = MV2Block(
            in_channels = channels[1],
            out_channels = channels[2],
            stride = 2,
            expansion_factor = expansion_factor
        )
        self.mv2_3 = MV2Block(
            in_channels = channels[2],
            out_channels = channels[3],
            stride = 1,
            expansion_factor = expansion_factor
        )
        self.mv2_4 = MV2Block(
            in_channels = channels[2],
            out_channels = channels[3],
            stride = 1,
            expansion_factor = expansion_factor
        )
        self.mv2_5 = MV2Block(
            in_channels = channels[3],
            out_channels = channels[4],
            stride = 2,
            expansion_factor = expansion_factor
        )
        self.mv2_6 = MV2Block(
            in_channels = channels[5],
            out_channels = channels[6],
            stride = 2,
            expansion_factor = expansion_factor
        )
        self.mv2_7 = MV2Block(
            in_channels = channels[7],
            out_channels = channels[8],
            stride = 2,
            expansion_factor = expansion_factor
        )


        mvit_depths = [2, 4, 3]
        num_heads = 4
        self.mvit_1 = MobileViTBlock(
            in_channels = channels[5],
            out_channels = dims[0],
            kernel_size = kernel_size,
            num_blocks = mvit_depths[0],
            num_heads = num_heads,
            mlp_scale = 2,
            patch_size = patch_size,
            # optional
            qkv_bias = qkv_bias,
            attn_p = attn_p,
            proj_p = proj_p
        )
        self.mvit_2 = MobileViTBlock(
            in_channels = channels[7],
            out_channels = dims[1],
            kernel_size = kernel_size,
            num_blocks = mvit_depths[1],
            num_heads = num_heads,
            mlp_scale = 2,
            patch_size = patch_size,
            # optional
            qkv_bias = qkv_bias,
            attn_p = attn_p,
            proj_p = proj_p
        )
        self.mvit_3 = MobileViTBlock(
            in_channels = channels[9],
            out_channels = dims[2],
            kernel_size = kernel_size,
            num_blocks = mvit_depths[2],
            num_heads = num_heads,
            mlp_scale = 2,
            patch_size = patch_size,
            # optional
            qkv_bias = qkv_bias,
            attn_p = attn_p,
            proj_p = proj_p
        )

        self.conv2 = ConvBlock(
            in_channels = channels[-2],
            out_channels = channels[-1], # <= 960
            kernel_size = 1,
            stride = 1,
            padding = 0,
            is_norm = True,
            is_activation = True
        )
        self.avg_pool = T.nn.AdaptiveAvgPool2d((1, 1)) # global avg pool
        self.fc = T.nn.Linear(channels[-1], num_classes, bias=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2_1(x)

        x = self.mv2_2(x)
        x = self.mv2_3(x)
        x = self.mv2_4(x)

        x = self.mv2_5(x)
        x = self.mvit_1(x)

        x = self.mv2_6(x)
        x = self.mvit_2(x)

        x = self.mv2_7(x)
        x = self.mvit_3(x)
        x = self.conv2(x)

        x = self.avg_pool(x).view(-1, x.shape[1])
        x = self.fc(x)

        return x