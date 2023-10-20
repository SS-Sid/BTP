import torch
from torchvision.models import shufflenet_v2_x2_0

class ShuffleNet(torch.nn.Module):
    def __init__(
            self,
            num_labels,
            pretrained
    ):
        super().__init__()
        if pretrained:
            self.model = shufflenet_v2_x2_0(pretrained=True)
        else:
            self.model = shufflenet_v2_x2_0(pretrained=False)
        
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, num_labels),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # x is single channel input, therefore repeat to 3 channel. 
        x = x.repeat(1,3,1,1)
        x = self.model(x)
        return x
