from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import CaptionModel
from vocab import Vocabulary


IMAGE_SIZE = 224


def get_image_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def save_checkpoint(path: str, model: CaptionModel, vocab: Vocabulary, config: dict) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab.to_dict(),
        "config": config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device)
    vocab = Vocabulary.from_dict(checkpoint["vocab"])
    config = checkpoint.get("config", {})

    model = CaptionModel(
        embed_size=int(config.get("embed_size", 256)),
        hidden_size=int(config.get("hidden_size", 256)),
        vocab_size=len(vocab),
        pad_idx=vocab.stoi["<PAD>"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, config


def load_image(image_source, device: str = "cpu") -> torch.Tensor:
    transform = get_image_transform()

    if isinstance(image_source, (str, Path)):
        image = Image.open(image_source).convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")

    return transform(image).to(device)


def generate_caption(
    model: CaptionModel,
    image: torch.Tensor,
    vocab: Vocabulary,
    max_length: int = 20,
    device: str = "cpu",
) -> str:
    image = image.to(device)
    token_ids = model.generate(
        image=image,
        sos_idx=vocab.stoi["<SOS>"],
        eos_idx=vocab.stoi["<EOS>"],
        max_length=max_length,
    )
    caption = vocab.decode(token_ids)
    return caption if caption else "No caption generated."
