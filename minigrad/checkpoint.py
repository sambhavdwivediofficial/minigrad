from pathlib import Path
import numpy as np
from .nn import Module
from .optim import Optimizer


def save_checkpoint(model: Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **model.state_dict())


def load_checkpoint(model: Module, path: str | Path) -> None:
    with np.load(Path(path), allow_pickle=False) as archive:
        model.load_state_dict({key: archive[key] for key in archive.files})


def save_training_state(model: Module, optimizer: Optimizer, path: str | Path, **metadata) -> None:
    combined = {}
    for name, value in model.state_dict().items():
        combined[f"model.{name}"] = value
    for name, value in optimizer.state_dict().items():
        combined[f"optimizer.{name}"] = np.asarray(value)
    for name, value in metadata.items():
        combined[f"meta.{name}"] = np.asarray(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **combined)


def load_training_state(model: Module, optimizer: Optimizer, path: str | Path) -> dict:
    model_state = {}
    optimizer_state = {}
    metadata = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        for key in archive.files:
            value = archive[key]
            if key.startswith("model."):
                model_state[key[len("model."):]] = value
            elif key.startswith("optimizer."):
                optimizer_state[key[len("optimizer."):]] = value
            elif key.startswith("meta."):
                metadata[key[len("meta."):]] = value
            else:
                raise ValueError(f"unrecognized key {key!r} in training-state checkpoint")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return metadata
