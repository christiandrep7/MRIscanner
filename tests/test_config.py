from pathlib import Path

from src.config import DataConfig, TrainConfig


def test_data_config_defaults():
    cfg = DataConfig()
    assert cfg.data_root == Path("data")
    assert cfg.train_dir_name == "Training"
    assert cfg.test_dir_name == "Testing"
    assert cfg.image_size == 224
    assert cfg.batch_size == 32
    assert cfg.num_workers == 2
    assert 0 < cfg.val_split < 1
    assert isinstance(cfg.seed, int)


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.architecture == "resnet50"
    assert cfg.epochs > 0
    assert cfg.learning_rate > 0
    assert cfg.early_stopping_patience > 0
    assert cfg.checkpoint_dir == Path("checkpoints")
    assert cfg.history_dir == Path("outputs")
    assert cfg.pretrained is True
    assert cfg.imagenet_weights_path is None


def test_train_config_paths_are_per_architecture():
    resnet_cfg = TrainConfig(architecture="resnet50")
    efficientnet_cfg = TrainConfig(architecture="efficientnet_b0")

    assert resnet_cfg.checkpoint_path == Path("checkpoints") / "resnet50_best_model.pth"
    assert efficientnet_cfg.checkpoint_path == Path("checkpoints") / "efficientnet_b0_best_model.pth"
    assert resnet_cfg.history_path == Path("outputs") / "resnet50_metrics_history.csv"
    assert resnet_cfg.checkpoint_path != efficientnet_cfg.checkpoint_path
    assert resnet_cfg.history_path != efficientnet_cfg.history_path


def test_configs_are_independently_overridable():
    cfg = DataConfig(batch_size=4, image_size=64)
    assert cfg.batch_size == 4
    assert cfg.image_size == 64
    # unrelated fields keep their defaults
    assert cfg.val_split == DataConfig().val_split
