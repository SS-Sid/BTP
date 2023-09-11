import torch
from sklearn.metrics import f1_score

def threshold(tensor, threshold_value=0.5):
    # Create a tensor of ones and zeros based on the threshold condition
    result = torch.where(tensor > threshold_value, torch.ones_like(tensor), torch.zeros_like(tensor))
    return result

class F1Score(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.reset()

    def forward(self, y_probs, y_true):
        y_preds = threshold(y_probs)
        
        self.predicted_labels = torch.cat((self.predicted_labels, y_preds.detach().cpu()))
        self.true_labels = torch.cat((self.true_labels, y_true.detach().cpu()))
        

    def get_metric(self):
        # when there is no true label 1 for a class, 
        # zero-division occurs which by default returns 0
        return f1_score(
            self.predicted_labels,
            self.true_labels,
            average=None,   # class-wise f1 scores
            zero_division=0 
        )


    def reset(self):
        self.true_labels = torch.zeros(0, self.num_labels)
        self.predicted_labels = torch.zeros(0, self.num_labels)