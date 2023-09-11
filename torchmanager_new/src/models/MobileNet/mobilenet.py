import torch
from torchvision.models import mobilenet_v2


class MobileNet(torch.nn.Module):
    def __init__(
            self,
            num_labels,
            pretrained
    ):
        super().__init__()
        if pretrained:
            self.model = mobilenet_v2(weights="DEFAULT")
        else:
            self.model = mobilenet_v2(weights=None)
        
        self.model.classifier[1] = torch.nn.Linear(1280, num_labels)
        self.model.classifier.add_module('2', torch.nn.Sigmoid())

    def forward(self, x):
        # x is single channel input, therefore repeat to 3 channel. 
        x = x.repeat(1,3,1,1)
        x = self.model(x)
        return x