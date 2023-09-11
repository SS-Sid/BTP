import torch
from torchvision.models import squeezenet1_0


class SqueezeNet(torch.nn.Module):
    def __init__(
            self,
            num_labels,
            pretrained
    ):
        super().__init__()
        if pretrained:
            self.model = squeezenet1_0(weights="DEFAULT")
        else:
            self.model = squeezenet1_0(weights=None)
        
        self.model.classifier[1] = torch.nn.Conv2d(
            512, 
            num_labels, 
            kernel_size=(1, 1), 
            stride=(1, 1)
        )
        self.model.classifier.add_module('4', torch.nn.Flatten())
        self.model.classifier.add_module('5', torch.nn.Sigmoid())

    def forward(self, x):
        # x is single channel input, therefore repeat to 3 channel. 
        x = x.repeat(1,3,1,1)
        x = self.model(x)
        return x
