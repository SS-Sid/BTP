import torch

def threshold(tensor, threshold_value=0.5):
    # Create a tensor of ones and zeros based on the threshold condition
    result = torch.where(tensor > threshold_value, torch.ones_like(tensor), torch.zeros_like(tensor))
    return result

class ExactMatchAccuracy(torch.nn.Module):
    def __init__(self,num_labels=14):
        super().__init__()
        self.num_labels = num_labels
        self.reset()

    def forward(self, y_probs, y_true_label):
        # y_probs (torch.Tensor): (num_samples, num_classes)
        # y_true_label (torch.Tensor): (num_samples, 1)

        y_preds = threshold(y_probs)
        num_correct = torch.all(y_preds==y_true_label,dim = 1).sum().item()

        self.correct += num_correct
        self.total += y_preds.size(0)

        return num_correct / y_preds.size(0)

    def get_metric(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0