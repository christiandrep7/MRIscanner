import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from src.checkpoint_io import load_training_checkpoint
from src.config import DataConfig
from src.data import get_eval_transforms
from src.gradcam_utils import generate_gradcam_overlay, save_overlay
from src.model import build_resnet50_model, get_device

checkpoint = Path("checkpoints/best_model.pth")
device = get_device()
payload = load_training_checkpoint(checkpoint, map_location=device)
class_to_idx = payload["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}
glioma_idx = class_to_idx["glioma"]
num_classes = len(class_to_idx)

model = build_resnet50_model(num_classes=num_classes, pretrained=False)
model.load_state_dict(payload["model_state_dict"])
model.to(device).eval()

cfg = DataConfig()
test_dir = cfg.data_root / cfg.test_dir_name
test_eval = datasets.ImageFolder(test_dir, transform=get_eval_transforms(cfg.image_size))
test_raw = datasets.ImageFolder(test_dir)
loader = DataLoader(test_eval, batch_size=1, shuffle=False)

out_dir = Path("outputs/figures/fig4_wrong_glioma")
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
    if saved >= 3:
        break

print(f"\nDone! Saved {saved} images to: {out_dir}")