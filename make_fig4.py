from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from src.checkpoint_io import load_training_checkpoint
from src.config import DataConfig
from src.data import get_eval_transforms
from src.evaluate import load_model_checkpoint
from src.gradcam_utils import generate_gradcam_overlay, save_overlay
from src.model import get_device


def main(
    checkpoint: Path = Path("checkpoints/resnet50_best_model.pth"),
    data_config: DataConfig | None = None,
    out_dir: Path = Path("outputs/figures/fig4_wrong_glioma"),
    max_saved: int = 3,
) -> int:
    """Saves Grad-CAM overlays for up to `max_saved` glioma scans the model misclassified."""
    cfg = data_config or DataConfig()
    device = get_device()
    payload = load_training_checkpoint(checkpoint, map_location=device)
    class_to_idx = payload["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    glioma_idx = class_to_idx["glioma"]
    num_classes = len(class_to_idx)

    # Self-describing checkpoint: builds whichever architecture it was trained with.
    model = load_model_checkpoint(checkpoint, num_classes=num_classes, device=device)

    test_dir = cfg.data_root / cfg.test_dir_name
    test_eval = datasets.ImageFolder(test_dir, transform=get_eval_transforms(cfg.image_size))
    test_raw = datasets.ImageFolder(test_dir)
    loader = DataLoader(test_eval, batch_size=1, shuffle=False)

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            pred = int(torch.argmax(model(x.to(device)), dim=1).item())
        true = int(y.item())
        if true != glioma_idx or pred == glioma_idx:
            continue
        img_path = test_raw.samples[i][0]
        image = Image.open(img_path).convert("RGB")
        overlay = generate_gradcam_overlay(model, image, device=device)
        pred_name = idx_to_class[pred]
        save_overlay(overlay, out_dir / f"wrong_glioma_pred_{pred_name}_{saved+1}.jpg")
        print(f"Saved: true=glioma, predicted={pred_name}, file={img_path}")
        saved += 1
        if saved >= max_saved:
            break

    print(f"\nDone! Saved {saved} images to: {out_dir}")
    return saved


if __name__ == "__main__":
    main()
