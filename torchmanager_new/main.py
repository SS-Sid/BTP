from src.data.data_pipeline import DataPipeline
from src.models.model_factory import ModelFactory
from src.training.trainer.classifier_trainer import ClassifierTrainer

import os


CONFIG_DIR = "/workspace/data/torchmanager/configs"
EXP = "exp1"
DATA = "nih_cxr_14_data_no_aug.yaml"
MODEL = "model_svit3.yaml"
TRAINER = "training.yaml"

data_config_path = os.path.join(CONFIG_DIR, EXP, DATA)
data_pipeline = DataPipeline(data_config_path)

train_dataloader = data_pipeline.data_loaders["train"]
val_dataloader = data_pipeline.data_loaders["val"]
test_dataloader = data_pipeline.data_loaders["test"]

model_config_path = os.path.join(CONFIG_DIR, EXP, MODEL)
model_factory = ModelFactory(model_config_path)

model = model_factory.model

training_config_path = os.path.join(CONFIG_DIR, EXP, TRAINER)
training_manager = ClassifierTrainer(training_config_path)

# make dataloaders dict with only train and val
train_dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader,
    'test': test_dataloader
}

train_history = training_manager.fit(model, train_dataloaders)

test_history = training_manager.test(model, test_dataloader)