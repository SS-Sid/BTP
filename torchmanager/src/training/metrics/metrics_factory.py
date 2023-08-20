import torch

class TopKAccuracy(torch.nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.correct = 0
        self.total = 0

    def forward(self, y_probs, y_true_label):
        # y_probs (torch.Tensor): (num_samples, num_classes)
        # y_true_label (torch.Tensor): (num_samples, 1)

        y_probs = torch.nn.functional.softmax(y_probs, dim=1)
        _, topk_preds = torch.topk(y_probs, self.k, dim=1)
        # topk_preds: (num_samples, k)

        y_true_label = y_true_label.view(-1, 1).expand_as(topk_preds)
        # y_true_label: (num_samples, k)

        num_correct = torch.eq(topk_preds, y_true_label).sum().item()

        self.correct += num_correct
        self.total += y_probs.size(0)

        return num_correct / y_probs.size(0)

    def get_metric(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0


def threshold(tensor, threshold_value=0.5):
    # Create a tensor of ones and zeros based on the threshold condition
    result = torch.where(tensor > threshold_value, torch.ones_like(tensor), torch.zeros_like(tensor))
    return result

class multiLabelAccuracy(torch.nn.Module):
    def __init__(self,num_labels=14):
        super().__init__()
        self.num_labels = num_labels
        self.correct = 0
        self.total = 0

    def forward(self, y_probs, y_true_label):
        # y_probs (torch.Tensor): (num_samples, num_classes)
        # y_true_label (torch.Tensor): (num_samples, 1)

        # y_probs = torch.nn.functional.softmax(y_probs, dim=1)
        # _, topk_preds = torch.topk(y_preds, self.num_labels, dim=1)
        # # topk_preds: (num_samples, k)

        # y_true_label = y_true_label.view(-1, 1).expand_as(topk_preds)
        # y_true_label: (num_samples, k)
        # print(y_preds.shape, y_true_label.shape)
        y_preds = threshold(y_probs)
        num_correct = torch.eq(y_preds, y_true_label).sum().item()

        self.correct += num_correct
        self.total += y_preds.size(0)*self.num_labels

        return num_correct / (y_preds.size(0)*self.num_labels)

    def get_metric(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0

class allLabelAccuracy(torch.nn.Module):
    def __init__(self,num_labels=14):
        super().__init__()
        self.num_labels = num_labels
        self.correct = 0
        self.total = 0

    def forward(self, y_probs, y_true_label):
        # y_probs (torch.Tensor): (num_samples, num_classes)
        # y_true_label (torch.Tensor): (num_samples, 1)

        # y_probs = torch.nn.functional.softmax(y_probs, dim=1)
        # _, topk_preds = torch.topk(y_preds, self.num_labels, dim=1)
        # # topk_preds: (num_samples, k)

        # y_true_label = y_true_label.view(-1, 1).expand_as(topk_preds)
        # y_true_label: (num_samples, k)
        # print(y_preds.shape, y_true_label.shape)
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