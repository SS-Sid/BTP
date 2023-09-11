# TODO: Add module docstring
from typing import Any, TypeVar

from torch.utils.data import Dataset, DataLoader

T_co = TypeVar('T_co', covariant=True)


def get_data_loader(
        dataset : Dataset[T_co],
        **kwargs : Any
) -> DataLoader[T_co]:
    # TODO: Add function docstring
    data_loader : DataLoader[T_co] = DataLoader(
        dataset,
        **kwargs
    )

    return data_loader
