import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
CSV_PATH  = BASE_DIR / "labels_labeled_split.csv"   # has columns: filepath, impacted_binary, split
DATA_ROOT = BASE_DIR / "Dental Xrays"               # folder that contains train/valid/test subfolders


class XrayDataset(Dataset):
    """
    Reads rows from CSV (optionally filtered by split), loads images.
    If CSV filepath doesn't exist, falls back to searching by filename under data_root.
    """
    def __init__(self, csv_path: Path, data_root: Path, split: str, transform=None):
        self.df = pd.read_csv(csv_path)

        required = {"filepath", "impacted_binary", "split"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}. Found: {list(self.df.columns)}")

        # Filter split
        self.df = self.df[self.df["split"] == split].copy().reset_index(drop=True)

        # Clean
        self.df = self.df.dropna(subset=["filepath", "impacted_binary"]).copy()
        self.df["impacted_binary"] = self.df["impacted_binary"].astype(int)

        self.transform = transform
        self.data_root = Path(data_root)

        # Build filename -> actual path index
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        name_to_path = {}
        for ext in exts:
            for p in self.data_root.rglob(ext):
                name_to_path[p.name] = p

        if len(name_to_path) == 0:
            raise FileNotFoundError(f"No images found under DATA_ROOT: {self.data_root}")

        self.name_to_path = name_to_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        raw = str(row["filepath"])
        p = Path(raw)

        # If CSV filepath doesn't exist, find by filename under DATA_ROOT
        if not p.exists():
            p = self.name_to_path.get(p.name)
            if p is None:
                raise FileNotFoundError(f"Could not find image:\nCSV said: {raw}\nSearched under: {self.data_root}")

        y = float(row["impacted_binary"])
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        fname = Path(str(row["filepath"])).name
        p = self.name_to_path.get(fname)
        if p is None:
            raise FileNotFoundError(f"Missing image file: {fname}")


        return img, torch.tensor(y, dtype=torch.float32)

# -----------------------------
# Metrics / evaluation helpers
# -----------------------------
@torch.no_grad()
def eval_loss(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(1)
        loss = loss_fn(logits, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    tp=fp=tn=fn=0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        probs = torch.sigmoid(model(x).squeeze(1))
        pred = (probs >= threshold).float()

        tp += ((pred==1) & (y==1)).sum().item()
        fp += ((pred==1) & (y==0)).sum().item()
        tn += ((pred==0) & (y==0)).sum().item()
        fn += ((pred==0) & (y==1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = (2 * precision * recall) / max(precision + recall, 1e-9)
    acc       = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

def threshold_sweep(model, loader, device, thresholds=None, require_recall=None):
    if thresholds is None:
        thresholds = [i/100 for i in range(5, 96, 5)]  # 0.05..0.95

    best_f1 = (-1, None, None)     # (f1, thresh, metrics)
    best_prec_given_recall = (-1, None, None)

    for t in thresholds:
        m = evaluate(model, loader, device, threshold=t)

        if m["f1"] > best_f1[0]:
            best_f1 = (m["f1"], t, m)

        if require_recall is not None and m["recall"] >= require_recall:
            if m["precision"] > best_prec_given_recall[0]:
                best_prec_given_recall = (m["precision"], t, m)

    return best_f1, best_prec_given_recall

# -----------------------------
# Main
# -----------------------------
def main():
    # Transforms for pretrained ResNet
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # Datasets/loaders
    train_ds = XrayDataset(CSV_PATH, DATA_ROOT, split="train", transform=tfm)
    val_ds   = XrayDataset(CSV_PATH, DATA_ROOT, split="val",   transform=tfm)
    test_ds  = XrayDataset(CSV_PATH, DATA_ROOT, split="test",  transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0)

    # Print split info
    full = pd.read_csv(CSV_PATH)
    print("\nSplit counts:\n", full["split"].value_counts())
    print("\nPos rate by split:\n", full.groupby("split")["impacted_binary"].mean())

    # pos_weight from TRAIN only
    train_df = train_ds.df
    pos = int(train_df["impacted_binary"].sum())
    neg = int((train_df["impacted_binary"] == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    print(f"\nTrain pos={pos} neg={neg} pos_weight={pos_weight.item():.3f}")

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # Loss + optimizer
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train
    best_val = float("inf")
    for epoch in range(1, 6):
        model.train()
        total = 0.0
        n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = total / max(n, 1)
        val_loss   = eval_loss(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "binary_impacted_best.pt")
            print("✅ Saved best model")

    # Load best + evaluate
    model.load_state_dict(torch.load("binary_impacted_best.pt", map_location=device))

    print("\nVAL metrics @0.50:", evaluate(model, val_loader, device, threshold=0.5))
    print("TEST metrics @0.50:", evaluate(model, test_loader, device, threshold=0.5))

    # Choose threshold on VAL (screening = prefer high recall)
    best_f1, best_prec_r90 = threshold_sweep(model, val_loader, device, require_recall=0.90)
    print("\n--- Threshold sweep on VAL ---")
    print("Best F1:", best_f1[1], best_f1[2])
    if best_prec_r90[1] is not None:
        print("Best Precision with Recall>=0.90:", best_prec_r90[1], best_prec_r90[2])
        chosen_t = best_prec_r90[1]
    else:
        # fallback: best F1
        chosen_t = best_f1[1]

    # Final report with chosen threshold
    val_m  = evaluate(model, val_loader, device, threshold=chosen_t)
    test_m = evaluate(model, test_loader, device, threshold=chosen_t)

    print(f"\nChosen screening threshold: {chosen_t:.2f}")
    print("FINAL VAL:", val_m)
    print("FINAL TEST:", test_m)

    with open("chosen_threshold.txt", "w") as f:
        f.write(str(chosen_t))

if __name__ == "__main__":
    main()
