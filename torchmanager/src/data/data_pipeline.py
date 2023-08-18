import yaml

from src.data.dataset_factory import create_dataset
from src.data.transforms_factory import create_transforms
from src.data.data_loader_factory import get_data_loader
from src.data.utils import get_validation_split, get_subset

from src.utils.logger import logging

from typing import Optional, Any, TypeVar, Generic
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.transforms import Compose  # type: ignore[import]

T_co = TypeVar('T_co', covariant=True)


class DataPipeline(Generic[T_co]):
    def __init__(
            self, 
            data_config_path: str,
    ) -> None:
        self.data_config_path = data_config_path

        self.data_config = self._load_data_config()
        self.datasets = self._generate_datasets()
        self.subsets = self._generate_subsets()
        self.transforms = self._generate_transforms()

        self._apply_transforms()

        self.data_loaders = self._generate_data_loaders()
        
    def _load_data_config(self) -> dict[str, dict[str, Any]]:
        with open(self.data_config_path, 'r') as data_config_file:
            data_config = yaml.load(data_config_file, Loader=yaml.FullLoader)
        
        logging.info(f"Data config loaded from {self.data_config_path}")
        logging.debug(f"Data config: {data_config}")
        
        return data_config
    
    def _generate_datasets(self) -> dict[str, Dataset[T_co] | Subset[T_co]]:
        dataset_common_config = self.data_config['dataset_common_config']
        datasets : dict[str, Dataset[T_co] | Subset[T_co]] = {}

        # if no dataset_split_config is provided
            # then create train dataset
        # else create datasets for each split in dataset_split_config
        if self.data_config.get('dataset_split_config') is None:
            logging.info("No dataset split config provided. Creating train dataset.")
            
            datasets['train'] = create_dataset(
                **dataset_common_config
            )
        else:
            logging.info("Dataset split config provided. Creating datasets for each split.")
            
            dataset_split_config = self.data_config['dataset_split_config']
            for data_split, split_config in dataset_split_config.items():
                datasets[data_split] = create_dataset(
                    **{
                        **dataset_common_config,
                        **split_config
                    }
                )
        
        # create remaining datasets using dataset_split_ratios
        if self.data_config.get('split_ratios') is not None:
            logging.info("Dataset split ratios provided. Creating datasets for each split ratio.")
            
            for data_split, split_config in self.data_config['split_ratios'].items():
                parent_data_split : str = split_config['parent']
                split_ratio : float = split_config['ratio']

                datasets[parent_data_split], datasets[data_split] = get_validation_split(
                    datasets[parent_data_split],
                    split_ratio
                )
        
        logging.info("Datasets created.")
        logging.debug(f"Datasets: {datasets}")
        
        return datasets
        
    def _generate_subsets(self) -> Optional[dict[str, Subset[T_co]]]:
        # if no subset_sizes is provided
            # then skip subset creation
        # else create subsets for each split in subset_sizes
        if self.data_config.get('subset_sizes') is None:
            logging.info("No subset sizes provided. Skipping subset creation.")
            
            return None
        else:
            logging.info("Subset sizes provided. Creating subsets for each split.")
            
            subsets : dict[str, Subset[T_co]] = {}
            for data_split, subset_size in self.data_config['subset_sizes'].items():
                subsets[data_split] = get_subset(
                    self.datasets[data_split],
                    subset_size
                )
            
            logging.info("Subsets created.")
            logging.debug(f"Subsets: {subsets}")
            
            return subsets

    def _generate_transforms(self) -> Optional[dict[str, Compose]]:
        # if no transforms_config is provided
            # then skip transform creation
        # else create transforms for each split in transforms_config
        if self.data_config.get('transforms_config') is None:
            logging.info("No transforms config provided. Skipping transform creation.")
            
            return None
        else:
            logging.info("Transforms config provided. Creating transforms for each split.")
            
            transforms : dict[str, Compose] = {}
            for data_split, transforms_config in self.data_config['transforms_config'].items():
                transforms[data_split] = create_transforms(
                    **transforms_config
                )
            
            logging.info("Transforms created.")
            logging.debug(f"Transforms: {transforms}")
            
            return transforms
    
    def _apply_transforms(self) -> None:
        # if no transforms are provided
            # then skip transform application
        # else if subset exists for data_split
                # then apply transforms to subset
            # else apply transforms to dataset
        if self.transforms is not None:
            logging.info("Applying transforms to datasets.")
            
            for data_split, transforms in self.transforms.items():
                if self.subsets is not None and self.subsets.get(data_split) is not None:
                    logging.info(f"Applying transforms to subset {data_split}.")
                    
                    self.subsets[data_split].dataset.transform = transforms # type: ignore[attr-defined]
                else:
                    # if dataset is a subset
                        # then apply transforms to subset
                    # else apply transforms to dataset
                    if isinstance(self.datasets[data_split], Subset):
                        logging.info(f"Applying transforms to subset {data_split}.")
                        
                        self.datasets[data_split].dataset.transform = transforms    # type: ignore[attr-defined]
                    else:
                        logging.info(f"Applying transforms to dataset {data_split}.")
                        
                        self.datasets[data_split].transform = transforms # type: ignore[attr-defined]
            
            logging.info("Transforms applied.")

    def _generate_data_loaders(self) -> dict[str, DataLoader[T_co]]:
        # make dataloader for each data split in datasets
        # if subset DNE for data split
            # then make dataloader for dataset
        data_loaders : dict[str, DataLoader[T_co]] = {}
        for data_split, dataset in self.datasets.items():
            if self.subsets is None or self.subsets.get(data_split) is None:
                logging.info(f"Creating data loader for dataset {data_split}.")
                
                data_loaders[data_split] = get_data_loader(
                    dataset,
                    **self.data_config['data_loader_config'].get(data_split)
                )
                
                logging.info(f"Data loader for dataset {data_split} created.")
            else:
                logging.info(f"Creating data loader for subset {data_split}.")
                
                data_loaders[data_split] = get_data_loader(
                    self.subsets[data_split],
                    **self.data_config['data_loader_config'].get(data_split)
                )
                
                logging.info(f"Data loader for subset {data_split} created.")
        
        logging.info("Data loaders created.")
        
        return data_loaders
