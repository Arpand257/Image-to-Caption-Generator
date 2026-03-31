import re
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold: int = 2) -> None:
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {token: index for index, token in self.itos.items()}

    def __len__(self) -> int:
        return len(self.itos)

    def tokenizer(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", str(text).lower())

    def build_vocab(self, sentence_list: list[str]) -> None:
        frequencies = Counter()
        next_index = len(self.itos)

        for sentence in sentence_list:
            for token in self.tokenizer(sentence):
                frequencies[token] += 1
                if frequencies[token] == self.freq_threshold and token not in self.stoi:
                    self.stoi[token] = next_index
                    self.itos[next_index] = token
                    next_index += 1

    def numericalize(self, text: str) -> list[int]:
        unk_index = self.stoi["<UNK>"]
        return [self.stoi.get(token, unk_index) for token in self.tokenizer(text)]

    def decode(self, token_ids: list[int]) -> str:
        special_tokens = {"<PAD>", "<SOS>", "<EOS>"}
        words = [self.itos.get(token_id, "<UNK>") for token_id in token_ids]
        words = [word for word in words if word not in special_tokens]
        return " ".join(words).strip()

    def to_dict(self) -> dict:
        return {
            "freq_threshold": self.freq_threshold,
            "stoi": self.stoi,
            "itos": self.itos,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        vocab = cls(freq_threshold=int(data.get("freq_threshold", 2)))
        vocab.stoi = {str(key): int(value) for key, value in data["stoi"].items()}
        vocab.itos = {int(key): str(value) for key, value in data["itos"].items()}
        return vocab
