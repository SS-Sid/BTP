from src.data.data_pipeline import DataPipeline
from src.models.model_factory import ModelFactory
from src.training.base_trainer import BaseTrainer

import os

import torch
import numpy as np
import random
# from torch import profiler
# import

# SEED = 85
# def seed_everything(seed):
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
    
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# seed_everything(SEED)

CONFIG_DIR = "/workspace/data/torchmanager/configs"
DATA = "nih_cxr_14_data_no_aug.yaml"
MODEL = "model_svit3.yaml"
TRAINER = "training.yaml"

data_config_path = os.path.join(CONFIG_DIR, DATA)
data_pipeline = DataPipeline(data_config_path)

train_dataloader, val_dataloader, test_dataloader = data_pipeline.data_loaders.values()

model_config_path = os.path.join(CONFIG_DIR, MODEL)
model_factory = ModelFactory(model_config_path)

model = model_factory.model

print(model)

# print total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

class ClassifierTrainer(BaseTrainer):
    def __init__(self, train_config_path):
        super().__init__(train_config_path)
    
    def train_step(self, batch_data, batch_idx):
        # init_batch
        
        # with profiler.profile(
        #     activities=[
        #         profiler.ProfilerActivity.CPU,
        #         profiler.ProfilerActivity.CUDA
        #     ],
        #     profile_memory=True,
        #     with_flops=True,
        #     with_stack=True,
        # ) as torch_profiler:
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
        #     torch_profiler.step()
        # torch_profiler.export_chrome_trace(f"workspace/data/torchmanager/logs/trace_{batch_idx}.json")

        # print("trace generated")
        # sys.stdout.flush()

        return results
    
    def val_step(self, batch_data, batch_idx):
        return self.train_step(batch_data, batch_idx)
    
    def test_step(self, batch_data, batch_idx):
        return self.train_step(batch_data, batch_idx)

training_config_path = os.path.join(CONFIG_DIR, TRAINER)
training_manager = ClassifierTrainer(training_config_path)

# make dataloaders dict with only train and val
train_dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader
}

train_history = training_manager.fit(model, train_dataloaders)

training_manager.test(model, test_dataloader)