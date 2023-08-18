import yaml

from src.training.losses import loss_factory
from src.training.optimizers import optimizer_factory
from src.training.metrics import metrics_factory

from src.training.early_stopper import EarlyStopper
from src.training.wandb_profiler import WandbProfiler
from src.utils.logger import logging

import torch

from typing import Any


class TrainManager:
    def __init__(self, train_config_path):
        self.train_config_path = train_config_path
        
        self.train_config = self._load_train_config()
        self.device = self._load_device()

        self.loss = None
        self.optimizer = None

        self._init_plugins()
    
    def _init_plugins(self):
        self.metrics = None
        self.early_stopper = None
        self.profiler = None
    
    # public methods
    def train_step(self, batch_data, model):
        # init_batch
        inputs, targets = batch_data

        # process_batch
        outputs = model(inputs)

        # compute_loss
        loss_value = self.loss(outputs, targets)

        # compute_metrics
        metrics_values = {}
        for metric_name, metric in self.metrics.items():
            metric_value = metric(outputs, targets)
            metrics_values[metric_name] = metric_value
        
        # end_batch
        results = metrics_values.update({
            'loss' : loss_value
        })

        return results

    def val_step(self, batch_data, model):
        return self.train_step(batch_data, model)
    
    def test_step(self, batch_data, model):
        return self.train_step(batch_data, model)
    
    
    def fit(self, model, dataloaders):
        model = self._move_to_device(model, self.device)
        self._load_loss()
        self._load_optimizer(model)
        self._load_metrics()

        self._load_early_stopper()
        self._load_profiler()

        _phases = dataloaders.keys()
        history = self._init_history(_phases)
        
        # if self.profiler is not None:
        #     self.profiler.watch(model, self.loss)

        for epoch in range(1, self.train_config['epochs']+1):
            for phase in _phases:
                dataloader = dataloaders[phase]

                if phase == 'train':
                    strategy = self.train_step
                    model.train()
                    torch.set_grad_enabled(True)
                else:
                    strategy = self.val_step
                    model.eval()
                    torch.set_grad_enabled(False)
                
                self._reset_metrics()

                running_loss = 0.0
                for batch_idx, batch_data in enumerate(dataloader):
                    batch_data = self._move_to_device(batch_data, self.device)
                    
                    if phase == 'train':
                        self.optimizer.zero_grad()

                    results = strategy(batch_data, model)

                    running_loss += results['loss'].item()
                    if phase == 'train':
                        results['loss'].backward()
                        self.optimizer.step()

                running_loss /= len(dataloader)
                    
                history[phase] = self._update_history(
                    history[phase], 
                    running_loss, 
                    epoch, 
                    phase
                )

            if self.early_stopper is not None:
                if self.early_stopper.check_stop(history):
                    break
        
        if self.profiler is not None:
            self.profiler.end()
        return history
    
    def test(self, model, dataloader):
        model = self._move_to_device(model, self.device)
        self._load_loss()
        self._load_metrics()

        model.eval()
        torch.set_grad_enabled(False)
        phase = 'test'
        history = self._init_history([phase])

        self._reset_metrics()

        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = self._move_to_device(batch_data, self.device)

            results = self.test_step(batch_data, model)

        history[phase] = self._update_history(history[phase], results, 1, phase)
        
        return history

    
    # private methods
    def _load_train_config(self) -> dict[str, Any]:
        with open(self.train_config_path, 'r') as train_config_file:
            train_config = yaml.load(train_config_file, Loader=yaml.FullLoader)
        logging.info(f"Train config loaded from {self.train_config_path}")
        logging.debug(f"Train config: {train_config}")
        return train_config
    
    def _load_device(self) -> torch.device:
        device = torch.device(self.train_config['device'])
        logging.info(f"Device loaded: {device}")
        return device

    def _load_loss(self) -> None:
        loss_config = self.train_config['loss']
        loss_name = loss_config['name']
        loss_params = loss_config['params']
        self.loss = loss_factory.get_loss_function(loss_config)

        logging.info(f"loss loaded: {loss_name}")
        logging.debug(f"loss params: {loss_params}")
    
    def _load_optimizer(self, model) -> None:
        optimizer_config = self.train_config['optimizer']
        optimizer_name = optimizer_config['name']
        optimizer_params = optimizer_config['params']
        self.optimizer = optimizer_factory.get_optimizer(optimizer_config, model)

        logging.info(f"Optimizer loaded: {optimizer_name}")
        logging.debug(f"Optimizer params: {optimizer_params}")
    
    def _load_metrics(self) -> None:
        metrics_config = self.train_config['metrics']
        self.metrics = {}
        for metric_name, metric_params in metrics_config.items():
            self.metrics[metric_name] = getattr(metrics_factory, metric_name)(**metric_params)
        
        logging.info(f"Metrics loaded: {self.metrics.keys()}")
        logging.debug(f"Metrics params: {self.metrics.values()}")
    
    def _reset_metrics(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
    
    def _init_history(self, _phases) -> dict[str, Any]:
        history = {}
        for phase in _phases:
            phase_history = {'loss' : [], 'metrics' : {}}
            phase_history['metrics'] = {metric_name : [] for metric_name in self.metrics.keys()}
            history[phase] = phase_history
        return history
    
    def _update_history(self, history, running_loss, epoch, phase) -> dict[str, Any]:
        profiler_dict = {}
        profiler_dict[f"{phase} metrics/loss"] = running_loss

        history['loss'].append(running_loss)
        for metric_name, metric in self.metrics.items():
            metric_value = metric.get_metric()
            history['metrics'][metric_name].append(metric_value)

            profiler_dict[f"{phase} metrics/{metric_name}"] = metric_value

        if self.profiler is not None:
            self.profiler.update(profiler_dict, epoch)
        
        logging.info(f"Epoch {epoch}:: {phase} Loss: {running_loss}\nMetrics: {profiler_dict}")
        return history


    def _move_to_device(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, torch.nn.Module):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {key: self._move_to_device(value, device) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._move_to_device(item, device) for item in obj]
        else:
            raise ValueError("Unsupported type. The input object must be a tensor, model, dictionary, or list.")
        
    def _load_early_stopper(self):
        if self.train_config.get('early_stopper') is not None:
            early_stopper_config = self.train_config['early_stopper']
            self.early_stopper = EarlyStopper(**early_stopper_config)
            logging.info(f"Early stopper loaded: {early_stopper_config}")
    
    def _load_profiler(self):
        if self.train_config.get('profiler') is not None:
            profiler_config = self.train_config['profiler']
            all_configs = self.train_config
            profiler_config['config'] = all_configs
            self.profiler = WandbProfiler(**profiler_config)
            logging.info(f"Profiler loaded: {profiler_config}")