# TODO: Add module docstring
from typing import Any, Dict

from src.data import transforms_zoo

from torchvision.transforms import Compose  # type: ignore[import]


def create_transforms(
        **kwargs : Dict[str, Any]
) -> Compose:
    # TODO: Add function docstring
    transforms : list[Any] = []
    for key, value in kwargs.items():
        if hasattr(transforms_zoo, key):
            transform_class : type[Any] = getattr(transforms_zoo, key)
        elif hasattr(transforms_zoo.transforms, key):
            transform_class : type[Any] = getattr(transforms_zoo.transforms, key)
        transform : Any = transform_class(**value)
        transforms.append(transform)
    composed_transforms : Compose = Compose(transforms)

    return composed_transforms
