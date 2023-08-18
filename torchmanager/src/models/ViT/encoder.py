import torch as T

from src.models.ViT.mhsa import MHSA
from src.models.ViT.mlp import MLP

"""
layernorm before each block and
residual connection after each block
=> norm->mhsa->res->norm->mlp->res
"""
class EncoderBlock(T.nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        attn_dim,
        mlp_scale,
        # optional
        qkv_bias=True,
        attn_p=0.,
        proj_p=0.
    ):
        super(EncoderBlock, self).__init__()

        hidden_features = int(attn_dim * mlp_scale)

        self.layer_norm_1 = T.nn.LayerNorm(attn_dim, eps=1e-6)
        # self.mhsa = MHSA(
        #     num_heads=num_heads,
        #     embed_dim=embed_dim,
        #     attn_dim=attn_dim,
        #     # optional
        #     qkv_bias=qkv_bias,
        #     attn_p=attn_p,
        #     proj_p=proj_p
        # )
        self.mhsa = T.nn.MultiheadAttention(
            num_heads = num_heads,
            embed_dim = embed_dim,
            batch_first = True
        )
        self.attn_dropout = T.nn.Dropout(p=proj_p)

        self.layer_norm_2 = T.nn.LayerNorm(attn_dim, eps=1e-6)
        self.mlp = MLP(
            in_features=attn_dim,
            hidden_features=hidden_features,
            out_features=attn_dim,
            # optional
            p=proj_p
        )


    def forward(self, X):
        X = self.layer_norm_1(X)
        # A = self.mhsa(X)
        A, _ = self.mhsa(X, X, X, need_weights=False)
        A = self.attn_dropout(A)

        A = self.layer_norm_2(A + X)
        Z = self.mlp(A)
        Z = Z + A

        return Z



class Encoder(T.nn.Module):
    def __init__(
        self,
        num_patches,
        num_blocks,
        num_heads,
        embed_dim,
        mlp_scale,
        # optional
        qkv_bias,
        attn_p,
        proj_p
    ):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks

        self.pos_embed = T.nn.Parameter(
            T.empty(
                1,
                num_patches + 1,
                embed_dim
            ).normal_(std=0.02) # from BERT
        )
        self.pos_dropout = T.nn.Dropout(p=proj_p)


        self.encoder = T.nn.ModuleList(
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
            for _ in range(self.num_blocks)
        )
        self.encoder_norm = T.nn.LayerNorm(embed_dim)


    def forward(self, X):
        X = X + self.pos_embed
        X = self.pos_dropout(X)

        Z = X
        for i in range(self.num_blocks):
            Z = self.encoder[i](Z)
        Z = self.encoder_norm(Z)

        return Z