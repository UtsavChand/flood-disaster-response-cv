import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Subset

from src.dataset import FloodDataset
from src.model import build_model

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
RES_DIR = ROOT / "results"
CKPT_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


class BCEDiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")

    def forward(self, logits, target):
        return self.bce(logits, target) + self.dice(logits, target)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, tp, fp, fn = 0.0, 0.0, 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * x.size(0)
        pred = (torch.sigmoid(logits) > 0.5).float()
        tp += (pred * y).sum().item()
        fp += (pred * (1 - y)).sum().item()
        fn += ((1 - pred) * y).sum().item()
    eps = 1e-7
    return {
        "loss": total_loss / len(loader.dataset),
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }


def train(model_name="unet", epochs=35, lr=3e-4, batch_size=8, subset=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = FloodDataset("train", train=True)
    val_ds = FloodDataset("val", train=False)
    if subset:  # smoke test on a few images
        train_ds = Subset(train_ds, range(subset))
        val_ds = Subset(val_ds, range(min(subset, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(model_name).to(device)
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_iou, history = 0.0, []
    best_path = CKPT_DIR / f"{model_name}_best.pth"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()

        train_loss = running / len(train_loader.dataset)
        val = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        **{f"val_{k}": v for k, v in val.items()}})

        saved = ""
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save(model.state_dict(), best_path)
            saved = " <- saved best"

        print(f"Epoch {epoch:02d}/{epochs} | train {train_loss:.4f} | "
              f"val loss {val['loss']:.4f} | val IoU {val['iou']:.4f} | "
              f"Dice {val['dice']:.4f} | {time.time() - t0:.0f}s{saved}")

    # save history and curves for the report
    df = pd.DataFrame(history)
    df.to_csv(RES_DIR / f"{model_name}_history.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df.epoch, df.train_loss, label="train")
    ax[0].plot(df.epoch, df.val_loss, label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(df.epoch, df.val_iou, color="green")
    ax[1].set_title("Validation IoU"); ax[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(RES_DIR / f"{model_name}_curves.png", dpi=150)
    plt.close()

    print(f"Done. Best val IoU: {best_iou:.4f} -> {best_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unet", choices=["unet", "deeplab"])
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--subset", type=int, default=None)
    a = p.parse_args()
    train(a.model, a.epochs, a.lr, a.batch_size, a.subset)