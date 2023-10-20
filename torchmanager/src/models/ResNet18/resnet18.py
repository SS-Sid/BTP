import torch
from torchvision.models import resnet18


class ResNet18(torch.nn.Module):
    def __init__(
            self,
            num_labels,
            pretrained
    ):
        super().__init__()
        if pretrained:
            self.model = resnet18(weights="DEFAULT")
        else:
            self.model = resnet18(weights=None)
        
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, num_labels),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # x is single channel input, therefore repeat to 3 channel. 
        x = x.repeat(1,3,1,1)
        x = self.model(x)
        return x
