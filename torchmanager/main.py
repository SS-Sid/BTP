from src.data.data_pipeline import DataPipeline
from src.models.model_factory import ModelFactory
from src.training.trainer import TrainManager

import os


CONFIG_DIR = "D:\\Work\\College\\Classroom\\torchmanager\\configs"
DATA_CONFIG_FILENAME = "data.yaml"
MODEL_CONFIG_FILENAME = "model.yaml"
TRAINING_CONFIG_FILENAME = "training.yaml"


data_config_path = os.path.join(CONFIG_DIR, DATA_CONFIG_FILENAME)
data_pipeline = DataPipeline(data_config_path)
train_dataloader, val_dataloader, test_dataloader = data_pipeline.data_loaders.values()

model_config_path = os.path.join(CONFIG_DIR, MODEL_CONFIG_FILENAME)
model_factory = ModelFactory(model_config_path)
model = model_factory.model

training_config_path = os.path.join(CONFIG_DIR, TRAINING_CONFIG_FILENAME)
training_manager = TrainManager(training_config_path)
train_dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader
}
fit_history = training_manager.fit(model, train_dataloaders)
