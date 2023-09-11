import importlib


def get_metric(metric_name, **metric_config):
    try:
        module_name = f'src.training.metrics.{metric_name}'
        module = importlib.import_module(module_name)
    except:
        try:
            module_name = f'torchmetrics'
            module = importlib.import_module(module_name)
            metric_class = getattr(module, metric_name)
        except:
            raise Exception(f"Could not find loss function {metric_name}")
        else:
            metric = metric_class(**metric_config)
            return metric
    else:
        metric_class = getattr(module, metric_name)
        metric = metric_class(**metric_config)
        return metric
