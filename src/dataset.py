from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_DIR = DATA_DIR / "Image"
MASK_DIR = DATA_DIR / "Mask"
SPLIT_CSV = DATA_DIR / "split.csv"

IMG_SIZE = 256
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_transforms(train):
    if train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class FloodDataset(Dataset):
    def __init__(self, split, train=False):
        df = pd.read_csv(SPLIT_CSV)
        self.stems = df[df.split == split].stem.astype(str).tolist()
        self.img_paths = {p.stem: p for p in IMAGE_DIR.glob("*")}
        self.mask_paths = {p.stem: p for p in MASK_DIR.glob("*")}
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        s = self.stems[i]
        img = np.array(Image.open(self.img_paths[s]).convert("RGB"))
        msk = (np.array(Image.open(self.mask_paths[s]).convert("L")) > 127).astype(np.uint8)

        if msk.shape[:2] != img.shape[:2]:
            raise ValueError(f"Size mismatch for {s}: image {img.shape[:2]} vs mask {msk.shape[:2]}")

        out = self.tf(image=img, mask=msk)
        return out["image"], out["mask"].unsqueeze(0).float()