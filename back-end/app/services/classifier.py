from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class PredictionResult:
    label: int
    label_name: str
    score: float
    message: str


class TextMemeClassifier:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.mode = "rule"
        self.sample_count = 0
        self.metrics: dict[str, float] = {}
        self.model: Pipeline | None = None
        self.label_mapping = {
            0: "正常文本",
            1: "疑似有害文本",
        }
        self.risk_keywords = [
            "仇恨",
            "歧视",
            "低等",
            "滚出去",
            "垃圾",
            "恶心",
            "侮辱",
            "种族主义",
            "不配",
            "贱",
            "该死",
            "废物",
            "杂种",
        ]

    def initialize(self) -> None:
        try:
            dataset = self._load_dataset()
            self.sample_count = len(dataset)
            self._train_model(dataset)
            self.mode = "ml"
        except Exception:
            self.mode = "rule"
            self.model = None

    def predict(self, text: str) -> PredictionResult:
        clean_text = text.strip()
        if not clean_text:
            return PredictionResult(
                label=0,
                label_name=self.label_mapping[0],
                score=0.0,
                message="输入为空，无法完成预测。",
            )

        if self.model is None:
            return self._predict_with_rules(clean_text)

        probabilities = self.model.predict_proba([clean_text])[0]
        model_score = float(probabilities[1])
        hit_count = self._count_risk_keywords(clean_text)

        # 为了让演示系统对明显攻击性表达更敏感，这里叠加一个关键词风险增强。
        if hit_count > 0:
            score = max(model_score, min(0.62 + 0.08 * hit_count, 0.95))
            label = 1
            message = "模型结果已结合风险关键词进行增强判断，建议进一步人工审核。"
        else:
            score = model_score
            label = int(score >= 0.5)
            if label == 1:
                message = "模型已完成预测，可结合人工判断进一步审核。"
            else:
                message = "模型认为该文本风险较低，可作为正常文本处理。"

        return PredictionResult(
            label=label,
            label_name=self.label_mapping[label],
            score=round(score, 4),
            message=message,
        )

    def _load_dataset(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"dataset not found: {self.data_path}")

        encodings = ("utf-8", "utf-8-sig", "gb18030", "gbk")
        last_error: Exception | None = None

        for encoding in encodings:
            try:
                dataset = pd.read_csv(self.data_path, encoding=encoding)
                break
            except Exception as error:  # pragma: no cover - fallback path
                last_error = error
        else:
            raise RuntimeError("failed to load dataset") from last_error

        required_columns = {"label", "text"}
        if not required_columns.issubset(dataset.columns):
            raise ValueError("dataset must contain label and text columns")

        dataset = dataset.loc[:, ["label", "text"]].copy()
        dataset["text"] = dataset["text"].fillna("").astype(str).str.strip()
        dataset["label"] = pd.to_numeric(dataset["label"], errors="coerce")
        dataset = dataset.dropna(subset=["label"])
        dataset["label"] = dataset["label"].astype(int)
        dataset = dataset[dataset["text"] != ""]
        dataset = dataset[dataset["label"].isin([0, 1])]
        dataset = dataset.drop_duplicates(subset=["text"])

        if len(dataset) < 50:
            raise ValueError("dataset is too small for training")

        return dataset

    def _train_model(self, dataset: pd.DataFrame) -> None:
        train_df, test_df = train_test_split(
            dataset,
            test_size=0.2,
            random_state=42,
            stratify=dataset["label"],
        )

        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(1, 3),
                        min_df=2,
                        lowercase=False,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        pipeline.fit(train_df["text"], train_df["label"])

        predictions = pipeline.predict(test_df["text"])
        self.metrics = {
            "accuracy": round(float(accuracy_score(test_df["label"], predictions)), 4),
            "precision": round(float(precision_score(test_df["label"], predictions)), 4),
            "recall": round(float(recall_score(test_df["label"], predictions)), 4),
            "f1": round(float(f1_score(test_df["label"], predictions)), 4),
        }
        self.model = pipeline

    def _predict_with_rules(self, text: str) -> PredictionResult:
        hit_count = self._count_risk_keywords(text)
        label = 1 if hit_count > 0 else 0
        score = min(0.58 + 0.08 * hit_count, 0.95) if label == 1 else 0.18
        message = (
            "当前使用规则模式返回结果，后续模型训练完成后会输出更稳定的预测。"
            if label == 1
            else "当前使用规则模式，文本未触发明显风险关键词。"
        )

        return PredictionResult(
            label=label,
            label_name=self.label_mapping[label],
            score=round(float(score), 4),
            message=message,
        )

    def _count_risk_keywords(self, text: str) -> int:
        return sum(1 for keyword in self.risk_keywords if keyword in text)
