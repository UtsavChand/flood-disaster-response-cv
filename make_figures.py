"""
make_figures.py - generates every figure used in README.md

Run from the PROJECT ROOT (the folder that contains data/, src/, results/):

    python make_figures.py

Creates:
    results/unet_curves.png
    results/figures/data_sample.png
    results/figures/mismatch_example.png
    results/figures/augmentation_samples.png
    results/figures/prediction_overlay.png
    results/figures/test_gt_vs_pred.png
and prints the test-set IoU / Dice so you can fill in the README.

Needs: data/Image, data/Mask, data/split.csv, results/unet_history.csv,
       checkpoints/unet_best.pth  (the last two are only needed for some figures).
Optional:  python make_figures.py --image 1234   (stem used for prediction_overlay.png)
"""
import argparse
import json
import traceback
from pathlib import Path

import albumentations as A
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(".")
IMG_DIR = ROOT / "data" / "Image"
MASK_DIR = ROOT / "data" / "Mask"
SPLIT_CSV = ROOT / "data" / "split.csv"
HIST_CSV = ROOT / "results" / "unet_history.csv"
CKPT = ROOT / "checkpoints" / "unet_best.pth"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 256
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------- helpers
def find_file(folder: Path, stem: str):
    for p in folder.glob(f"{stem}.*"):
        return p
    return None


def load_image(stem):
    p = find_file(IMG_DIR, stem)
    return np.array(Image.open(p).convert("RGB")) if p else None


def load_mask(stem):
    p = find_file(MASK_DIR, stem)
    if not p:
        return None
    return (np.array(Image.open(p).convert("L")) > 127).astype(np.uint8)


def overlay(img, mask, alpha=0.5):
    out = img.copy()
    m = mask == 1
    out[m] = ((1 - alpha) * img[m] + alpha * np.array([0, 0, 255])).astype(np.uint8)
    return out


def load_split():
    df = pd.read_csv(SPLIT_CSV)
    cols = {c.lower(): c for c in df.columns}
    split_col = next((cols[c] for c in ("split", "set", "subset", "partition") if c in cols), None)
    id_col = next((cols[c] for c in ("stem", "id", "name", "image", "filename", "file") if c in cols),
                  df.columns[0])
    if split_col is None:
        raise SystemExit(f"Could not find a split column in {SPLIT_CSV}. Columns: {list(df.columns)}")
    df["_stem"] = df[id_col].astype(str).map(lambda s: Path(s).stem)
    df["_split"] = df[split_col].astype(str).str.lower().replace({"valid": "val", "validation": "val"})
    return {s: df.loc[df["_split"] == s, "_stem"].tolist() for s in ("train", "val", "test")}


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def get_model():
    from src.predict import load_model

    model = load_model(str(CKPT), "unet")
    model.eval()
    return model


def forward_256(model, img):
    """Exactly the evaluation protocol: resize to 256, normalise, sigmoid."""
    import torch

    dev = next(model.parameters()).device
    x = A.Compose([A.Resize(SIZE, SIZE), A.Normalize(MEAN, STD)])(image=img)["image"]
    x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(dev)
    with torch.no_grad():
        return torch.sigmoid(model(x))[0, 0].cpu().numpy()


# ----------------------------------------------------------------- figures
def fig_curves():
    df = pd.read_csv(HIST_CSV)
    low = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            for k, v in low.items():
                if n in k:
                    return v
        return None

    ep = pick("epoch")
    x = df[ep] if ep else np.arange(1, len(df) + 1)
    tl, vl, vi = pick("train_loss", "train loss"), pick("val_loss", "val loss"), pick("val_iou", "iou")
    if vi is None:
        raise RuntimeError(f"No IoU column found. Columns: {list(df.columns)}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if tl:
        ax[0].plot(x, df[tl], label="train loss")
    if vl:
        ax[0].plot(x, df[vl], label="val loss")
    ax[0].set(title="Loss per epoch", xlabel="epoch", ylabel="BCE + Dice loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(x, df[vi], color="tab:green")
    b = int(df[vi].values.argmax())
    ax[1].scatter([x.iloc[b] if hasattr(x, "iloc") else x[b]], [df[vi].iloc[b]], color="red", zorder=3,
                  label=f"best {df[vi].iloc[b]:.3f}")
    ax[1].set(title="Validation IoU per epoch", xlabel="epoch", ylabel="IoU")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    save(fig, ROOT / "results" / "unet_curves.png")


def fig_data_sample(splits):
    stem = splits["train"][0]
    img, m = load_image(stem), load_mask(stem)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img); ax[0].set_title(f"Image {stem}")
    ax[1].imshow(m, cmap="gray"); ax[1].set_title("Ground-truth mask (white = flood)")
    for a in ax:
        a.axis("off")
    save(fig, FIG_DIR / "data_sample.png")


def fig_mismatch():
    stem = "14"
    ip, mp = find_file(IMG_DIR, stem), find_file(MASK_DIR, stem)
    if not (ip and mp):
        raise RuntimeError("Image/Mask 14 not found - edit `stem` in fig_mismatch() to another bad pair.")
    img, m = Image.open(ip).convert("RGB"), Image.open(mp).convert("L")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img); ax[0].set_title(f"Image {stem}: {img.size[0]}x{img.size[1]}")
    ax[1].imshow(m, cmap="gray"); ax[1].set_title(f"Mask {stem}: {m.size[0]}x{m.size[1]}  (size mismatch)")
    for a in ax:
        a.axis("off")
    save(fig, FIG_DIR / "mismatch_example.png")


def fig_augmentation(splits):
    rng = np.random.default_rng(0)
    stems = list(rng.choice(splits["train"], 4, replace=False))
    fig, ax = plt.subplots(4, 3, figsize=(9, 12))
    for r, stem in enumerate(stems):
        t = A.Compose([
            A.Resize(SIZE, SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=1.0 if r == 3 else 0.2),  # last row: forced vertical flip
            A.RandomBrightnessContrast(p=0.3),
        ])
        out = t(image=load_image(stem), mask=load_mask(stem))
        i, m = out["image"], out["mask"]
        for c, (pic, title, cmap) in enumerate([(i, "image", None), (m, "mask", "gray"), (overlay(i, m), "overlay", None)]):
            ax[r, c].imshow(pic, cmap=cmap)
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(title)
    save(fig, FIG_DIR / "augmentation_samples.png")


def fig_prediction(splits, stem):
    from src.predict import predict

    model = get_model()
    stem = stem or splits["test"][0]
    img = load_image(stem)
    mask = predict(model, Image.fromarray(img))
    pct = mask.mean() * 100
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    ax[0].imshow(img); ax[0].set_title("Original")
    ax[1].imshow(mask, cmap="gray"); ax[1].set_title("Predicted flood mask")
    ax[2].imshow(overlay(img, mask)); ax[2].set_title(f"Overlay - {pct:.1f}% flooded")
    for a in ax:
        a.axis("off")
    save(fig, FIG_DIR / "prediction_overlay.png")
    print(f"  image {stem}: {pct:.1f}% flooded  <- use this number in the README caption")


def eval_test_and_fig(splits):
    model = get_model()
    rows, TP, FP, FN = [], 0, 0, 0
    cache = {}
    for stem in splits["test"]:
        img, gt = load_image(stem), load_mask(stem)
        if img is None or gt is None or img.shape[:2] != gt.shape[:2]:
            print(f"  skipping {stem} (missing or mismatched)")
            continue
        p = forward_256(model, img)
        pred = (p > 0.5).astype(np.uint8)
        g = A.Resize(SIZE, SIZE)(image=img, mask=gt)
        gt256, img256 = g["mask"], g["image"]
        tp = int(((pred == 1) & (gt256 == 1)).sum())
        fp = int(((pred == 1) & (gt256 == 0)).sum())
        fn = int(((pred == 0) & (gt256 == 1)).sum())
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        rows.append((stem, tp / max(tp + fp + fn, 1)))
        cache[stem] = (img256, gt256, pred)

    iou = TP / (TP + FP + FN)
    dice = 2 * TP / (2 * TP + FP + FN)
    prec, rec = TP / (TP + FP), TP / (TP + FN)
    res = dict(n_images=len(rows), test_iou=iou, test_dice=dice, test_precision=prec, test_recall=rec)
    (ROOT / "results" / "unet_test_metrics.json").write_text(json.dumps(res, indent=2))
    print("\n  TEST SET (U-Net):", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}))

    # show the image whose own IoU is closest to the overall test IoU
    stem, s_iou = min(rows, key=lambda r: abs(r[1] - iou))
    img256, gt256, pred = cache[stem]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    ax[0].imshow(img256); ax[0].set_title(f"Image {stem}")
    ax[1].imshow(gt256, cmap="gray"); ax[1].set_title("Ground truth")
    ax[2].imshow(pred, cmap="gray"); ax[2].set_title(f"Prediction (IoU = {s_iou:.3f})")
    for a in ax:
        a.axis("off")
    save(fig, FIG_DIR / "test_gt_vs_pred.png")
    print(f"  figure image {stem}: IoU = {s_iou:.3f}  <- use this number in the README caption")


# ----------------------------------------------------------------- main
def step(name, fn, *args):
    print(f"[{name}]")
    try:
        fn(*args)
    except Exception:
        print(f"  FAILED:\n{traceback.format_exc()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="stem for prediction_overlay.png (default: first test image)")
    a = ap.parse_args()

    splits = load_split()
    print({k: len(v) for k, v in splits.items()})

    step("curves", fig_curves)
    step("data sample", fig_data_sample, splits)
    step("mismatch example", fig_mismatch)
    step("augmentation samples", fig_augmentation, splits)
    if CKPT.exists():
        step("prediction overlay", fig_prediction, splits, a.image)
        step("test metrics + gt vs pred", eval_test_and_fig, splits)
    else:
        print(f"\n{CKPT} not found - skipping the two model-based figures.")
    print("\nDone. Now: git add readme.md results/ && git commit && git push")
