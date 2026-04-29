from __future__ import annotations

from app.services.classifier import extract_meme_feature_vector
from bert.data import build_data_splits


class BertMemeFusionDataset:
    def __init__(self, df, tokenizer, max_length: int) -> None:
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, object]:
        text = self.texts[index]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = self.labels[index]
        encoded["meme_features"] = extract_meme_feature_vector(text)
        return encoded


__all__ = ["BertMemeFusionDataset", "build_data_splits"]
