import torch as T
import torch.nn.functional as F

class MHSA(T.nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        attn_dim,
        # optional
        qkv_bias=True,
        attn_p = 0.,
        proj_p = 0.
    ):
        super(MHSA, self).__init__()

        self.num_heads = num_heads # h
        self.attn_dim = attn_dim # c
        self.head_dim = self.attn_dim // self.num_heads # a

        self.Wqkv = T.nn.Linear(
            in_features=embed_dim,
            out_features=3*self.attn_dim,
            bias=qkv_bias
        )

        self.head_concat = T.nn.Linear(
            in_features=self.attn_dim,
            out_features=self.attn_dim
        )

        self.attn_dropout = T.nn.Dropout(attn_p)
        self.proj_dropout = T.nn.Dropout(proj_p)


    def forward(self, X):
        # X: (-1, n+1, d)
        # QKV: (-1, n+1, 3*c)
        QKV = self.Wqkv(X)
        # QKV: (-1, n+1, 3, h, a)
        QKV = QKV.view(
            QKV.size(0), QKV.size(1), 3, self.num_heads, self.head_dim
        )
        # QKV: (3, -1, h, n+1, a)
        QKV = QKV.permute(2, 0, 3, 1, 4)
        # Q, K, V: (-1, h, n+1, a)
        Q, K, V = QKV[0], QKV[1], QKV[2]


        # K_T: (-1, h, a, n+1)
        K_T = K.transpose(-2, -1)
        # score: (-1, h, n+1, n+1)
        score = Q @ K_T
        # S: (-1, h, n+1, n+1)
        S = F.softmax(score / (K.size(-1) ** 0.5), dim=-1)
        S = self.attn_dropout(S)

        # Z: (-1, h, n+1, a)
        Z = S @ V
        # Z: (-1, n+1, h, a)
        Z = Z.transpose(1, 2)
        # Z: (-1, n+1, h * a)
        # Z: (-1, n+1, c)
        Z = Z.flatten(2)

        # A: (-1, n+1, c)
        A = self.head_concat(Z)
        A = self.proj_dropout(A)

        return A