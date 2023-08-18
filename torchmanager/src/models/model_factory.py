from typing import Any

import yaml
import importlib

from src.utils.logger import logging

class ModelFactory:
    def __init__(self, model_config_path):
        self.model_config_path = model_config_path

        self.model_config = self._load_model_config()
        self.model = self._load_model()

    def _load_model_config(self) -> dict[str, Any]:
        with open(self.model_config_path, 'r') as model_config_file:
            model_config = yaml.load(model_config_file, Loader=yaml.FullLoader)
        logging.info(f"Model config loaded from {self.model_config_path}")
        logging.debug(f"Model config: {model_config}")
        return model_config

    def _load_model(self):
        model_name = self.model_config['name']
        module_name = f'src.models.{model_name}.{model_name.lower()}'
        module = importlib.import_module(module_name)
        model_class = getattr(module, model_name)
        model_params = self.model_config['params']
        model = model_class(**model_params)
        logging.info(f"Model loaded: {model_name}")
        logging.debug(f"Model params: {model_params}")
        return model
