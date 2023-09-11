import torch as T

class MLP(T.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        p = 0.
    ):
        super(MLP, self).__init__()

        self.mlp = T.nn.Sequential(
            T.nn.Linear(
                in_features=in_features,
                out_features=hidden_features
            ),
            T.nn.GELU(),
            T.nn.Dropout(p),

            T.nn.Linear(
                in_features=hidden_features,
                out_features=out_features
            ),
            T.nn.Dropout(p)
        )


    def forward(self, x):
        output = self.mlp(x)

        return output