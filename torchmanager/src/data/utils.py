# TODO: Add module docstring
from typing import Sequence, TypeVar

from torch.utils.data import Dataset, Subset, random_split

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


def get_validation_split(
        dataset : Dataset[T], 
        split_ratio : float = 0.2
) -> tuple[Subset[T], Subset[T]]:
    # TODO: Add function docstring
    train_size = int((1. - split_ratio) * len(dataset)) # type: ignore[arg-type, operator]
    val_size = len(dataset) - train_size    # type: ignore[arg-type]
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size]
    )

    return train_dataset, val_dataset


def get_subset(
        dataset : Dataset[T_co],
        subset_size : int
) -> Subset[T_co]:
    # TODO: Add function docstring
    indices : Sequence[int] = list(range(subset_size))
    subset : Subset[T_co] = Subset(dataset, indices)

    return subset