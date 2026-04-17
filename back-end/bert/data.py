from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from app.services.classifier import load_dataset


@dataclass
class DataSplit:
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    test_df: pd.DataFrame


def build_data_splits(
    data_path,
    random_seed: int = 42,
    smoke_test: bool = False,
    train_sample_limit: int | None = None,
    eval_sample_limit: int | None = None,
) -> DataSplit:
    dataset = load_dataset(data_path)

    train_df, temp_df = train_test_split(
        dataset,
        test_size=0.2,
        random_state=random_seed,
        stratify=dataset["label"],
    )
    eval_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_seed,
        stratify=temp_df["label"],
    )

    if smoke_test:
        train_sample_limit = train_sample_limit or 256
        eval_sample_limit = eval_sample_limit or 128

    if train_sample_limit:
        train_df = _sample_dataframe(train_df, train_sample_limit, random_seed)
    if eval_sample_limit:
        eval_df = _sample_dataframe(eval_df, eval_sample_limit, random_seed)
        test_df = _sample_dataframe(test_df, eval_sample_limit, random_seed)

    return DataSplit(
        train_df=train_df.reset_index(drop=True),
        eval_df=eval_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )


def _sample_dataframe(df: pd.DataFrame, sample_limit: int, random_seed: int) -> pd.DataFrame:
    if len(df) <= sample_limit:
        return df
    sampled_parts = []
    label_counts = df["label"].value_counts(normalize=True).sort_index()

    for label, ratio in label_counts.items():
        label_df = df[df["label"] == label]
        label_n = max(1, round(sample_limit * float(ratio)))
        label_n = min(len(label_df), label_n)
        sampled_parts.append(label_df.sample(n=label_n, random_state=random_seed))

    sampled = pd.concat(sampled_parts, axis=0)
    if len(sampled) > sample_limit:
        sampled = sampled.sample(n=sample_limit, random_state=random_seed)
    return sampled


class BertTextDataset:
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, object]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = self.labels[index]
        return encoded
