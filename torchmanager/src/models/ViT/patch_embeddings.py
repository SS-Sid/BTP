import torch as T

"""
Conv Layer:
    patchifies image using same kernel
    and stride.

    out channels simulates linear projection
    to map n patches to vector size d.
"""
class PatchEmbedding(T.nn.Module):
    def __init__(
        self,
        in_channels, # c
        patch_size, # p
        embed_dim # d
    ):
        super(PatchEmbedding, self).__init__()

        self.conv = T.nn.Conv2d(
            in_channels=in_channels, # c
            out_channels=embed_dim, # d
            kernel_size=patch_size, # p
            stride=patch_size, # p
            padding=0
        )


    def forward(self, x):
        # x: (-1, c, h, w)
        # embed: (-1, d, h//p, w//p)
        embed = self.conv(x)

        # num_patches: n = h//p * w//p
        # X: (-1, d, n)
        X = embed.flatten(2)
        # X: (-1, n, d)
        X = X.transpose(1, 2)

        return X