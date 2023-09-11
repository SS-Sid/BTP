import torch
from sklearn.metrics import roc_auc_score


class MultiLabelAUROC(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.reset()
    
    def forward(self, y_probs, y_true):
        self.predicted_probs = torch.cat((self.predicted_probs, y_probs.detach().cpu()))
        self.true_labels = torch.cat((self.true_labels, y_true.detach().cpu()))
    
    def get_metric(self):
        auroc_scores = []
        for label_idx in range(self.num_labels):
            true_label_col = self.true_labels[:, label_idx]
            predicted_probs_col = self.predicted_probs[:, label_idx]
            auroc = roc_auc_score(true_label_col.numpy(), predicted_probs_col.numpy())
            auroc_scores.append(auroc)

        return auroc_scores
    
    def reset(self):
        self.true_labels = torch.zeros(0, self.num_labels)
        self.predicted_probs = torch.zeros(0, self.num_labels)