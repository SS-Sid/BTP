import yaml

from src.data.nih_preloader import NIH_CXR_14_Preloader
from src.data.transforms_factory import create_transforms

from torchvision.transforms import Compose  # type: ignore[import]
from typing import Dict


PRELOAD_CONFIG_FILE = "/workspace/data/torchmanager/configs/NIH_preloader/preloader_configs.yaml"

# Load data config yaml
with open(PRELOAD_CONFIG_FILE, 'r') as data_config_file:
    data_config = yaml.load(data_config_file, Loader=yaml.FullLoader)

# instantiate preloader variables
org_dir = data_config['root']
preloader_dir = data_config['destination']

transforms : Dict[str, Compose] = {}
for data_split, transforms_config in data_config['transforms_config'].items():
    transforms[data_split] = create_transforms(
        **transforms_config
    )


# preload the data
train_preloader = NIH_CXR_14_Preloader(
    root=org_dir,
    destination=preloader_dir,
    train=True,
    # batch_size=1000, # Default: None => all data in single batch
    transform=transforms['train']
)

test_preloader = NIH_CXR_14_Preloader(
    root=org_dir,
    destination=preloader_dir,
    train=False,
    # batch_size=1000, # Default: None => all data in single batch
    transform=transforms['test']
)
