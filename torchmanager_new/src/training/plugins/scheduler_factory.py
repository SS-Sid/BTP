import importlib

from typing import Any, Dict

import torch

def get_lr_scheduler(
    lr_scheduler_config: Dict[str, Any],
    optimizer: torch.optim.Optimizer
):
    lr_scheduler_name : str = lr_scheduler_config['name']
    try:
        module_name = f'torch.optim.lr_scheduler'
        module = importlib.import_module(module_name)
        lr_scheduler_class = getattr(module, lr_scheduler_name)
    except:
        raise Exception(f"Could not find lr_scheduler {lr_scheduler_name}")
    else:
        lr_scheduler_params = lr_scheduler_config['params']
        lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_params)
        return lr_scheduler