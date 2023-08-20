import importlib

from typing import Any, Dict


def get_loss_function(
    loss_config: Dict[str, Any]
):
    loss_function_name : str = loss_config['name']
    try:
        module_name = f'src.training.losses.{loss_function_name}'
        module = importlib.import_module(module_name)
    except:
        try:
            module_name = f'torch.nn.modules.loss'
            module = importlib.import_module(module_name)
            loss_function_class = getattr(module, loss_function_name)
        except:
            raise Exception(f"Could not find loss function {loss_function_name}")
        else:
            loss_function_params = loss_config['params']
            loss_function = loss_function_class(**loss_function_params)
            return loss_function
    else:
        loss_function_class = getattr(module, loss_function_name)
        loss_function_params = loss_config['params']
        loss_function = loss_function_class(**loss_function_params)
        return loss_function
