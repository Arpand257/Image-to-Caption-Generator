import os
import random

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CaptionDataset, collate_fn
from model import CaptionModel
from utils import get_image_transform, save_checkpoint
from vocab import Vocabulary


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = os.environ.get("IMAGE_DIR", "image")
CAPTIONS_FILE = os.environ.get("CAPTIONS_FILE", "captions.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pth")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
EMBED_SIZE = int(os.environ.get("EMBED_SIZE", "256"))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
FREQ_THRESHOLD = int(os.environ.get("FREQ_THRESHOLD", "2"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
SEED = int(os.environ.get("SEED", "42"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vocab(captions_file: str, freq_threshold: int) -> Vocabulary:
    df = pd.read_csv(captions_file)
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_vocab(df["caption"].astype(str).tolist())
    return vocab


def train() -> None:
    set_seed(SEED)

    vocab = build_vocab(CAPTIONS_FILE, FREQ_THRESHOLD)
    dataset = CaptionDataset(
        root_dir=IMAGE_DIR,
        captions_file=CAPTIONS_FILE,
        vocab=vocab,
        transform=get_image_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = CaptionModel(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=len(vocab),
        pad_idx=vocab.stoi["<PAD>"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch_index, (images, captions) in enumerate(loader, start=1):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            optimizer.zero_grad()
            logits = model(images, inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batch_count += 1

            if MAX_BATCHES and batch_index >= MAX_BATCHES:
                break

        average_loss = running_loss / max(batch_count, 1)
        print(f"Epoch {epoch}/{EPOCHS} - loss: {average_loss:.4f}")

    save_checkpoint(
        MODEL_PATH,
        model,
        vocab,
        {
            "embed_size": EMBED_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "image_dir": IMAGE_DIR,
            "captions_file": CAPTIONS_FILE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "freq_threshold": FREQ_THRESHOLD,
        },
    )
    print(f"Saved checkpoint to {MODEL_PATH}")


if __name__ == "__main__":
    train()
