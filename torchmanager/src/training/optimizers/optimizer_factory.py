import importlib

from typing import Any, Dict

import torch


def get_optimizer(
    optimizer_config: Dict[str, Any],
    model: torch.nn.Module
):
    optimizer_name : str = optimizer_config['name']
    try:
        module_name = f'torch.optim'
        module = importlib.import_module(module_name)
        optimizer_class = getattr(module, optimizer_name)
    except:
        raise Exception(f"Could not find optimizer {optimizer_name}")
    else:
        optimizer_params = optimizer_config['params']
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        return optimizer
