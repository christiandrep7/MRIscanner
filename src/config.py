from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    data_root: Path = Path("data")
    train_dir_name: str = "Training"
    test_dir_name: str = "Testing"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    val_split: float = 0.15
    seed: int = 42


@dataclass
class TrainConfig:
    epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    checkpoint_dir: Path = Path("checkpoints")
    history_path: Path = Path("outputs") / "metrics_history.csv"
    # ImageNet backbone: False = random init (slow); True = pretrained or local file / hub cache
    pretrained: bool = True
    imagenet_weights_path: Path | None = None
