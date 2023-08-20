# TODO: Add module docstring
import os
from typing import Any
from pathlib import Path

from src.data import dataset_zoo

import torchvision  # type: ignore

VisionDatasetAlias = torchvision.datasets.vision.VisionDataset


def get_torchvision_datasets(
        dataset_name: str,
        root: Path,
        **kwargs : Any
) -> VisionDatasetAlias:
    # TODO: Add function docstring
    dataset_class : type[VisionDatasetAlias] = getattr(
        torchvision.datasets,
        dataset_name
    )
    dataset : VisionDatasetAlias = dataset_class(
        root,
        **kwargs
    )
    return dataset


def get_local_datasets(
        root: Path,
        **kwargs : Any
) -> VisionDatasetAlias:
    # TODO: Add function docstring
    dataset : VisionDatasetAlias = get_torchvision_datasets(
        dataset_name='ImageFolder',
        root=root,
        **kwargs
    )
    return dataset


def get_custom_datasets(
        dataset_name: str,
        root: Path,
        **kwargs : Any
) -> VisionDatasetAlias:
    if hasattr(dataset_zoo, dataset_name):
        dataset_class : type[VisionDatasetAlias] = getattr(
            dataset_zoo, 
            dataset_name
        )
        dataset : VisionDatasetAlias = dataset_class(
            root=root,
            **kwargs
        )
        return dataset


def create_dataset(
        import_type: str,
        dataset_name: str,
        root: Path,
        **kwargs : Any
) -> VisionDatasetAlias:
    # TODO: Add function docstring
    if import_type == 'torchvision':
        dataset = get_torchvision_datasets(
            dataset_name=dataset_name,
            root=root,
            **kwargs
        )
    elif import_type == 'local':
        dataset = get_local_datasets(
            root=root,
            **kwargs
        )
    elif import_type == 'custom':
        dataset = get_custom_datasets(
            dataset_name=dataset_name,
            root=root,
            **kwargs
        )
    return dataset  # type: ignore # will be fixed using exception handling
