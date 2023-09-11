from src.training.trainer.base_trainer import BaseTrainer

class ClassifierTrainer(BaseTrainer):
    def __init__(self, train_config_path):
        super().__init__(train_config_path)
    
    def train_step(self, batch_data, batch_idx):
        # init_batch
        inputs, targets = batch_data

        # process_batch
        outputs = self.model(inputs)

        # compute_loss
        loss_value = self.loss(outputs, targets)

        # compute_metrics
        metrics_values = {}
        for metric_name, metric in self.metrics.items():
            metric_value = metric(outputs, targets)
            metrics_values[metric_name] = metric_value
        
        # end_batch
        results = {
            'loss' : loss_value,
            **metrics_values
        }

        return results
    
    def val_step(self, batch_data, batch_idx):
        return self.train_step(batch_data, batch_idx)
    
    def test_step(self, batch_data, batch_idx):
        return self.train_step(batch_data, batch_idx)