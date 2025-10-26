import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIR = Path("real_captchas")
OUTPUT_CSV = Path("submission.csv")

CHECKPOINTS_DIR = Path("checkpoints")
TS_PATH = CHECKPOINTS_DIR / "ctc_ocr_ts.pt"
SD_PATH = CHECKPOINTS_DIR / "ctc_ocr.pt"
CHARSET_JSON = CHECKPOINTS_DIR / "charset.json"

IMG_W, IMG_H = 192, 96

from image_processing import PreprocConfig, preprocess_pipeline, load_rgb

class CRNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=2,
                           bidirectional=True, batch_first=False, dropout=0.1)
        self.head = nn.Linear(512, n_classes)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.mean(dim=2)
        feat = feat.permute(2,0,1)
        out, _ = self.rnn(feat)
        logits = self.head(out)
        return logits

def load_charset(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j["idx_to_char"], int(j["blank_idx"])

def greedy_decode(logits: torch.Tensor, blank_idx: int) -> List[int]:
    preds = logits.argmax(dim=-1)[:, 0].detach().cpu().tolist()
    out, prev = [], None
    for p in preds:
        if p != blank_idx and p != prev:
            out.append(p)
        prev = p
    return out

def indices_to_string(indices: List[int], idx_to_char: List[str], blank_idx: int) -> str:
    return "".join(idx_to_char[i] for i in indices if i != blank_idx)

def preprocess_for_model(img_path: Path) -> np.ndarray:
    cfg = PreprocConfig()
    rgb = load_rgb(img_path)
    _, _, bw_clean, _, _ = preprocess_pipeline(rgb, cfg)
    if bw_clean.shape != (IMG_H, IMG_W):
        bw_clean = np.array(Image.fromarray(bw_clean).resize((IMG_W, IMG_H), Image.NEAREST), dtype=np.uint8)
    return bw_clean

def to_model_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)

def hexish_sort_key(name: str):
    order = {c:i for i, c in enumerate("0123456789abcdef")}
    return [order.get(ch, 100 + ord(ch)) for ch in name.lower()]

def main():
    if not INPUT_DIR.exists():
        return

    files = sorted([p for p in INPUT_DIR.glob("*.png")], key=lambda p: hexish_sort_key(p.name))
    if not files:
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="\n") as f:
            f.write("hstaffor\n")
        return

    idx_to_char, blank_idx = load_charset(CHARSET_JSON)
    n_classes = len(idx_to_char)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if TS_PATH.exists():
        model = torch.jit.load(str(TS_PATH), map_location=device)
        model.eval()
    else:
        model = CRNN(n_classes).to(device)
        sd = torch.load(str(SD_PATH), map_location=device)
        state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        model.load_state_dict(state)
        model.eval()

    total = len(files)   # NEW
    print(f"Processing {total} images...\n")   # NEW

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="\n") as f:
        f.write("hstaffor\n")
        for i, p in enumerate(files, start=1):
            try:
                arr = preprocess_for_model(p)
                x = to_model_tensor(arr, device)
                with torch.no_grad():
                    logits = model(x)
                    log_probs = F.log_softmax(logits, dim=-1)
                seq_idx = greedy_decode(log_probs, blank_idx)
                pred = indices_to_string(seq_idx, idx_to_char, blank_idx)
            except Exception:
                pred = ""

            f.write(f"{p.name},{pred}\n")

            if i % 50 == 0 or i == total:   # NEW (progress print every 50)
                print(f"[{i}/{total}] {p.name} → {pred}")

if __name__ == "__main__":
    main()
