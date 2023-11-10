from src.data.data_pipeline import DataPipeline
from src.models.model_factory import ModelFactory
from src.training.trainer.classifier_trainer import ClassifierTrainer
from torchsummary import summary
import os


CONFIG_DIR = "/home/karan/Documents/GitHub/BTP/torchmanager/configs"
EXP = "exp8"
DATA = "nih_cxr_14_data_no_aug.yaml"#
MODEL = "model_mobilevit.yaml"
TRAINER = "training.yaml"

# data_config_path = os.path.join(CONFIG_DIR, EXP, DATA)
# data_pipeline = DataPipeline(data_config_path)

# train_dataloader = data_pipeline.data_loaders["train"]
# val_dataloader = data_pipeline.data_loaders["val"]
# test_dataloader = data_pipeline.data_loaders["test"]
    
print("Model Loading")
model_config_path = os.path.join(CONFIG_DIR, EXP, MODEL)
model_factory = ModelFactory(model_config_path)

model = model_factory.model
print("Model Loaded")
print(summary(model.cuda(),(3,224,224)))
print(model.cpu())
# print("Classifier Loading")

# training_config_path = os.path.join(CONFIG_DIR, EXP, TRAINER)
# training_manager = ClassifierTrainer(training_config_path)

# # make dataloaders dict with only train and val
# train_dataloaders = {
#     'train': train_dataloader,
#     'val': val_dataloader,
#     'test': test_dataloader
# }
# print("Classifier Loaded")
# print("Training Start")
# train_history = training_manager.fit(model, train_dataloaders)

# print("Training Done")

# test_history = training_manager.test(model, test_dataloader)