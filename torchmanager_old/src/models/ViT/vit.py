import torch as T
from src.models.ViT.patch_embeddings import PatchEmbedding
from src.models.ViT.encoder import Encoder


class ViT(T.nn.Module):
    def __init__(
        self,
        img_size,
        num_classes,
        patch_size,
        embed_dim,
        num_blocks,
        num_heads,
        mlp_scale,
        # optional
        qkv_bias = True,
        attn_p = 0.,
        proj_p = 0.
    ):
        super(ViT, self).__init__()

        # INPUT -> EMBEDDING
        ci, hi, wi = img_size
        num_patches = (hi//patch_size) * (wi//patch_size)


        self.patch_embed = PatchEmbedding(
            in_channels=ci,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.cls_token = T.nn.Parameter(
            T.zeros(
                1,
                1,
                embed_dim
            )
        )

        # ENCODER BLOCKS
        self.encoder = Encoder(
            num_patches=num_patches,
            num_blocks=num_blocks,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_scale=mlp_scale,
            # optional
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=proj_p
        )

        # MLP HEAD
        self.mlp_head = T.nn.Linear(
            in_features=embed_dim,
            out_features=num_classes
        )


    def forward(self, x):
        # INPUT -> EMBEDDING
        X = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            X.size(0), -1, -1
        )
        X = T.cat((cls_token, X), dim=1)

        # ENCODER
        Z = self.encoder(X)
        classifier_tokens = Z[:, 0]

        # MLP HEAD
        output = self.mlp_head(classifier_tokens)

        return output