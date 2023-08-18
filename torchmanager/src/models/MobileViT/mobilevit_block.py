import torch as T
from src.models.MobileViT.conv_block import ConvBlock
from src.models.ViT.encoder import EncoderBlock

class MobileViTBlock(T.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_blocks,
        num_heads,
        mlp_scale,
        patch_size,
        # optional
        qkv_bias,
        attn_p,
        proj_p
    ):
        super(MobileViTBlock, self).__init__()

        self.ph, self.pw = patch_size

        # local representations
        # nxn conv
        self.conv1 = ConvBlock(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = 1,
            is_norm = True,
            is_activation = True
        )
        # 1x1 conv
        self.conv2 = ConvBlock(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            is_norm = False,
            is_activation = False
        )

        # global representations
        embed_dim = out_channels
        encoder_layers = [
            EncoderBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                attn_dim=embed_dim,
                mlp_scale=mlp_scale,
                # optional
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=proj_p
            )
            for _ in range(num_blocks)
        ]
        encoder_layers.append(T.nn.LayerNorm(embed_dim))
        self.encoder = T.nn.Sequential(*encoder_layers)

        # fusion
        # 1x1 conv
        self.conv3 = ConvBlock(
            in_channels = out_channels,
            out_channels = in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            is_norm = True,
            is_activation = True
        )
        # nxn conv
        self.conv4 = ConvBlock(
            in_channels = 2 * in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = 1,
            is_norm = True,
            is_activation = True
        )



    def forward(self, x):
        y = x.clone()

        # local representations
        x = self.conv1(x)
        x = self.conv2(x)

        ## feature map -> embedding
        x, nh, nw = self.unfold(x)
        # global representations
        x = self.encoder(x)
        ## embedding -> feature map
        x = self.fold(x, nh, nw)


        # fusion
        x = self.conv3(x)
        x = T.cat((x, y), 1)
        x = self.conv4(x)

        return x


    def unfold(self, x):
        ni, ic, ih, iw = x.size()

        nh = ih // self.ph
        nw = iw // self.pw

        # x: (ni, ic, ih, iw)
        # x: (ni, ic, nh, ph, nw, pw)
        x = x.view(ni, ic, nh, self.ph, nw, self.pw)
        # x: (ni, ph, pw, nh, nw, ic)
        x = x.permute(0, 3, 5, 2, 4, 1)
        # x: (ni * ph * pw, nh * nw, ic)
        x = x.contiguous().view(ni * self.ph * self.pw, nh * nw, ic)

        return x, nh, nw


    def fold(self, x, nh, nw):
        n, p, ic = x.size()

        ni = n // (self.ph * self.pw)
        ih = nh * self.ph
        iw = nw * self.pw

        # x: (ni * ph * pw, nh * nw, ic)
        # x: (ni, ph, pw, nh, nw, ic)
        x = x.view(ni, self.ph, self.pw, nh, nw, ic)
        # x: (ni, ic, nh, ph, nw, pw)
        x = x.permute(0, 5, 3, 1, 4, 2)
        # x: (ni, ic, nh, ih, iw)
        x = x.contiguous().view(ni, ic, ih, iw)

        return x