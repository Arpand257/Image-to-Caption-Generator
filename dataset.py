import os

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    def __init__(self, root_dir: str, captions_file: str, vocab, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab

        df = pd.read_csv(captions_file)
        df["image"] = df["image"].astype(str).str.strip()
        df["caption"] = df["caption"].astype(str).str.strip()
        df["image_path"] = df["image"].apply(lambda name: os.path.join(root_dir, name))
        self.df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

        if self.df.empty:
            raise ValueError("No valid image-caption pairs were found.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption_tokens = [self.vocab.stoi["<SOS>"]]
        caption_tokens.extend(self.vocab.numericalize(row["caption"]))
        caption_tokens.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(caption_tokens, dtype=torch.long)


def collate_fn(batch):
    images, captions = zip(*batch)
    image_batch = torch.stack(images)
    pad_idx = 0
    caption_batch = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
    return image_batch, caption_batch
