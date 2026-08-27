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
    architecture: str = "resnet50"
    epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    checkpoint_dir: Path = Path("checkpoints")
    history_dir: Path = Path("outputs")
    # ImageNet backbone: False = random init (slow); True = pretrained or local file / hub cache
    pretrained: bool = True
    imagenet_weights_path: Path | None = None

    @property
    def checkpoint_path(self) -> Path:
        """Per-architecture: training 3 models never overwrites each other's checkpoint."""
        return self.checkpoint_dir / f"{self.architecture}_best_model.pth"

    @property
    def history_path(self) -> Path:
        """Per-architecture: training 3 models never overwrites each other's history CSV."""
        return self.history_dir / f"{self.architecture}_metrics_history.csv"
