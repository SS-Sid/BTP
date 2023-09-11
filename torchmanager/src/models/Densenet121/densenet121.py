import torch
from torchvision.models import densenet121


class Densenet121(torch.nn.Module):
    def __init__(
            self,
            num_labels,
            pretrained
    ):
        super().__init__()
        if pretrained:
            self.model = densenet121(weights="DEFAULT")
        else:
            self.model = densenet121(weights=None)
        
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, num_labels),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # x is single channel input, therefore repeat to 3 channel. 
        x = x.repeat(1,3,1,1)
        x = self.model(x)
        return x
