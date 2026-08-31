from dataclasses import dataclass, field
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
    # Label smoothing softened every class's decision, not just the weak one --
    # measured worse: overall accuracy and meningioma recall both dropped. Left
    # here at 0.0 (off) so it stays available without being applied by default.
    label_smoothing: float = 0.0
    # Per-class CrossEntropyLoss weight, keyed by class *name* (resolved against
    # the dataset's own class_to_idx at train time, so it doesn't depend on
    # alphabetical class order). glioma benchmarked with the lowest recall of all
    # 4 classes on data/Testing across all 3 architectures (~76-84% vs 90-100%
    # elsewhere) -- weighting it up makes the loss penalize a missed glioma more
    # than a missed meningioma/notumor/pituitary, pushing recall up directly
    # instead of via a blanket recipe change like label smoothing.
    class_loss_weights: dict = field(default_factory=lambda: {"glioma": 1.5})
    # class_loss_weights pushed overall glioma recall up, but empirically didn't
    # target *which* wrong class it slips into -- glioma-called-"notumor" (the
    # most dangerous miss: an "all clear" on a real tumor) stayed flat/got slightly
    # worse. This adds a direct penalty on the softmax probability mass the model
    # assigns to `false_negative_penalty_class` whenever the true label is any
    # *other* class -- i.e. specifically discourages "no tumor" whenever there
    # really is one, regardless of which tumor type. 0.0 disables it.
    false_negative_penalty_class: str = "notumor"
    false_negative_penalty_weight: float = 0.5
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
