import torch
import torch.nn as nn


class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(128, embed_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoded = self.features(images).flatten(1)
        return self.projection(encoded)


class DecoderRNN(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, pad_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_hidden = nn.Linear(embed_size, hidden_size)

    def forward(self, image_features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(captions)
        hidden = self.init_hidden(image_features).unsqueeze(0)
        outputs, _ = self.rnn(embedded, hidden)
        return self.fc(outputs)


class CaptionModel(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, pad_idx: int) -> None:
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, pad_idx)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        image_features = self.encoder(images)
        return self.decoder(image_features, captions)

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_length: int = 20,
    ) -> list[int]:
        self.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image_features = self.encoder(image)
        hidden = self.decoder.init_hidden(image_features).unsqueeze(0)
        current = torch.tensor([[sos_idx]], device=image.device)
        tokens: list[int] = []

        for _ in range(max_length):
            embedded = self.decoder.embedding(current)
            output, hidden = self.decoder.rnn(embedded, hidden)
            logits = self.decoder.fc(output[:, -1, :])
            next_token = int(logits.argmax(dim=-1).item())

            if next_token == eos_idx:
                break

            tokens.append(next_token)
            current = torch.tensor([[next_token]], device=image.device)

        return tokens
