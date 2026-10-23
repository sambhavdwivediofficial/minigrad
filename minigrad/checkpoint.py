from pathlib import Path
import numpy as np
from .nn import Module


def save_checkpoint(model: Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **model.state_dict())


def load_checkpoint(model: Module, path: str | Path) -> None:
    with np.load(Path(path), allow_pickle=False) as archive:
        model.load_state_dict({key: archive[key] for key in archive.files})
