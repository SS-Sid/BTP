import yaml

from src.training.losses import loss_factory
from src.training.optimizers import optimizer_factory

from src.training.metrics import metrics_factory

from src.training.plugins.early_stopper import EarlyStopper
from src.training.plugins.wandb_profiler import WandbProfiler
from src.training.plugins.scheduler_factory import get_lr_scheduler
from src.training.plugins.checkpointer import Checkpointer

from src.utils.logger import logging

import torch
import sys
from typing import Any, Dict
# from torch import profiler


class BaseTrainer:
    def __init__(self, train_config_path):
        self.train_config_path = train_config_path
        self.train_config = self._load_train_config()

        self._init_variables()
    
    def _init_variables(self):
        self.device = self._load_device()
        self.model = None
        self.loss = None
        self.optimizer = None

        self._init_plugins()
    
    def _init_plugins(self):
        self.metrics = None
        self.early_stopper = None
        self.profiler = None
        self.lr_scheduler = None
        self.checkpointer = None
    
    # public methods
    def train_step(self, batch_data, batch_idx):
        ...

    def val_step(self, batch_data, batch_idx):
        ...
    
    def test_step(self, batch_data, batch_idx):
        ...
    

    def fit(self, model, dataloaders):
        self.on_start_fit(model, dataloaders)
        
        for epoch in range(self.start_epoch+1, self.train_config['epochs']+1):
            if self.is_training_stopped:
                break

            self.on_epoch_start()
            
            for phase in self._phases:
                self.on_start_phase(phase)

                for batch_idx, batch_data in enumerate(self.dataloader):
                    self.on_start_iteration(phase, batch_data)

                    results = self.strategy(batch_data, batch_idx)

                    self.on_end_iteration(phase, results)
                
                self.on_end_phase(phase, epoch)
            
            self.on_end_epoch(epoch)
        
        self.on_end_fit()

        return self.history

    def on_start_fit(self, model, dataloaders):
        self.model = self._move_to_device(model, self.device)
        self._load_loss()
        self._load_optimizer()

        self._load_metrics()
        self._load_early_stopper()
        self._load_profiler()
        self._load_lr_scheduler()
        self._load_checkpointer()

        self._phases = dataloaders.keys()
        self.history = self._init_history(self._phases)
        self.start_epoch = 0
        self.is_training_stopped = False

        if self.checkpointer is not None:
            if self.checkpointer.resume == True:
                self.start_epoch = self.checkpointer.load(
                    checkpoint_path=self.train_config['checkpointer']['resume_path'],
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    history=self.history
                )
                logging.info(f"Resuming training from epoch {self.start_epoch}")
    
    def on_start_epoch():
        pass

    def on_start_phase(self, phase):
        self.dataloader = self.dataloaders[phase]
        
        if phase == 'train':
            self.strategy = self.train_step
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.strategy = self.val_step
            self.model.eval()
            torch.set_grad_enabled(False)
                
        self._reset_metrics()
    
    def on_start_iteration(self, phase, batch_data):
        self.batch_data = self._move_to_device(batch_data, self.device)
        
        if phase == 'train':
            self.optimizer.zero_grad()
    
    def on_end_iteration(self, phase, results):
        if phase == 'train':
            results['loss'].backward()
            self.optimizer.step()
    
        self.running_loss += results['loss'].detach()
    
    def on_end_phase(self, phase, epoch):
        self.running_loss /= len(self.dataloader)
            
        self.history[phase] = self._update_history(
            self.history[phase],
            epoch,
            phase
        )

    def on_end_epoch(self, epoch):
        if self.checkpointer is not None:
            if epoch % self.checkpointer.save_period == 0:
                self.checkpointer.save(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    history=self.history
                )

        if self.early_stopper is not None:
            if self.early_stopper.check_stop(self.history):
                self.is_training_stopped = True

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
    
    def on_end_fit(self):
        if self.profiler is not None:
            self.profiler.end()
    
    def test(self, model, dataloader):
        self.model = self._move_to_device(model, self.device)
        self._load_loss()
        self._load_metrics()

        model.eval()
        torch.set_grad_enabled(False)
        phase = 'test'
        history = self._init_history([phase])

        self._reset_metrics()

        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = self._move_to_device(batch_data, self.device)

            results = self.test_step(batch_data, batch_idx)

            self.running_loss += results['loss'].detach()
        self.running_loss /= len(dataloader)
        
        history[phase] = self._update_history(history[phase], 1, phase)
        
        return history

    
    # private methods
    def _load_train_config(self) -> Dict[str, Any]:
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
    
    def _load_optimizer(self) -> None:
        optimizer_config = self.train_config['optimizer']
        optimizer_name = optimizer_config['name']
        optimizer_params = optimizer_config['params']
        self.optimizer = optimizer_factory.get_optimizer(optimizer_config, self.model)

        logging.info(f"Optimizer loaded: {optimizer_name}")
        logging.debug(f"Optimizer params: {optimizer_params}")
    
    def _load_metrics(self) -> None:
        metrics_config = self.train_config['metrics']
        self.metrics = {}
        for metric_name, metric_params in metrics_config.items():
            self.metrics[metric_name] = metrics_factory.get_metric(metric_name, **metric_params)
        
        logging.info(f"Metrics loaded: {self.metrics.keys()}")
        logging.debug(f"Metrics params: {self.metrics.values()}")
    
    def _reset_metrics(self) -> None:
        self.running_loss = 0.0
        for metric in self.metrics.values():
            metric.reset()

    def _init_history(self, _phases) -> Dict[str, Any]:
        history = {}
        for phase in _phases:
            phase_history = {'loss' : []}
            metrics_dict = {metric_name : [] for metric_name in self.metrics.keys()}
            phase_history = {**phase_history, **metrics_dict}
            
            history[phase] = phase_history
        return history
    
    def _update_history(self, history, epoch, phase) -> Dict[str, Any]:
        profiler_dict = {}
        profiler_dict[f"{phase}/loss"] = self.running_loss.item()

        history['loss'].append(self.running_loss)
        for metric_name, metric in self.metrics.items():
            metric_value = metric.get_metric()
            history[metric_name].append(metric_value)

            profiler_dict[f"{phase}/{metric_name}"] = metric_value

        if self.profiler is not None and phase != 'test':
            self.profiler.update(profiler_dict, epoch)
        
        logging.info(f"Epoch {epoch}/{self.train_config['epochs']} {phase}::\n{profiler_dict}")
        return history


    def _move_to_device(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, torch.nn.Module):
            return obj.to(device)
        elif isinstance(obj, Dict):
            return {key: self._move_to_device(value, device) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._move_to_device(item, device) for item in obj]
        elif isinstance(obj, tuple):
            return tuple([self._move_to_device(item, device) for item in obj])
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
    
    def _load_lr_scheduler(self):
        if self.train_config.get('lr_scheduler') is not None:
            lr_scheduler_config = self.train_config['lr_scheduler']
            self.lr_scheduler = get_lr_scheduler(lr_scheduler_config, self.optimizer)
            logging.info(f"LR Scheduler loaded: {lr_scheduler_config}")
    
    def _load_checkpointer(self):
        # if checkpointer config is given, then load checkpointer
        # resume option is also inside checkpointer config
        if self.train_config.get('checkpointer') is not None:
            checkpointer_config = self.train_config['checkpointer']
            self.checkpointer = Checkpointer(**checkpointer_config)
            logging.info(f"Checkpointer loaded: {checkpointer_config}")