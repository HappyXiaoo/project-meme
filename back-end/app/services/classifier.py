from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

LABEL_MAPPING = {
    0: "正常文本",
    1: "疑似有害文本",
}

DEFAULT_MODEL_TYPE = "meme_features_v1"
SUPPORTED_MODEL_TYPES = {"baseline", "meme_features_v1"}

RISK_KEYWORDS = [
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

GROUP_TERMS = [
    "他们",
    "这些人",
    "那群人",
    "某些人",
    "这个群体",
    "这种群体",
    "男人",
    "女人",
    "男的",
    "女的",
    "本地人",
    "外地人",
    "中国人",
    "外国人",
    "黑人",
    "白人",
    "黄种人",
    "穆斯林",
    "留学生",
    "网友",
]

INTENSIFIER_TERMS = [
    "太",
    "真",
    "真的",
    "非常",
    "特别",
    "简直",
    "根本",
    "天生",
]

SARCASM_MARKERS = [
    "呵呵",
    "笑死",
    "也是没谁了",
    "就这",
    "还搁这",
    "真有你的",
    "典中典",
    "某些人",
]

CONTRAST_MARKERS = [
    "我们",
    "他们",
    "这种人",
    "那种人",
    "一类人",
    "本地人",
    "外地人",
]

MEME_PHRASES = [
    "不是我说",
    "某些人",
    "果然",
    "建议把",
    "还搁这",
    "就这",
    "也是没谁了",
    "典中典",
]

MITIGATION_TERMS = [
    "反对",
    "抵制",
    "谴责",
    "批评",
    "避免",
    "禁止",
    "不应该",
    "不能",
    "不要",
    "应当避免",
    "应该避免",
]

REPORTING_TERMS = [
    "言论",
    "说法",
    "表述",
    "行为",
    "发言",
    "评论",
    "内容",
    "现象",
    "问题",
    "审核",
    "举报",
    "描述",
    "讨论",
]

ANALYSIS_TERMS = [
    "像是",
    "带有",
    "属于",
    "涉嫌",
    "存在",
    "可能",
    "会",
]

STRONG_ATTACK_PATTERNS = [
    re.compile(r"不配.{0,8}(尊重|活着|存在)"),
    re.compile(r"低人一等"),
    re.compile(r"滚出去"),
    re.compile(r"天生.{0,6}(差|贱|坏|低等)"),
    re.compile(r"(杂种|废物|垃圾|贱种)"),
]

TEMPLATE_PATTERNS = [
    re.compile(r"不是.{0,12}但是"),
    re.compile(r"我不是.{0,12}但是"),
    re.compile(r"不是我说"),
    re.compile(r"某些人"),
    re.compile(r"果然.{0,10}都"),
    re.compile(r"建议把.{0,10}都"),
]


@dataclass
class PredictionResult:
    label: int
    label_name: str
    score: float
    message: str


@dataclass
class ArtifactPaths:
    model_path: Path
    metrics_path: Path
    metadata_path: Path


class MemeExpressionFeatureExtractor(BaseEstimator, TransformerMixin):
    feature_names = [
        "template_pattern_hits",
        "group_reference_hits",
        "intensifier_hits",
        "sarcasm_hits",
        "contrast_hits",
        "meme_phrase_hits",
        "punctuation_burst_hits",
        "repeated_character_hits",
        "risk_group_cooccurrence",
        "mitigation_hits",
        "reporting_hits",
        "analysis_hits",
        "direct_attack_pattern_hits",
        "meme_group_cooccurrence",
    ]

    def fit(self, X: Any, y: Any = None) -> "MemeExpressionFeatureExtractor":
        return self

    def transform(self, X: Any) -> csr_matrix:
        rows = [self._extract_features(str(text)) for text in X]
        return csr_matrix(np.asarray(rows, dtype=float))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names, dtype=object)

    def _extract_features(self, text: str) -> list[float]:
        normalized = text.strip()
        group_hits = sum(normalized.count(term) for term in GROUP_TERMS)
        risk_hits = sum(normalized.count(term) for term in RISK_KEYWORDS)

        return [
            float(sum(bool(pattern.search(normalized)) for pattern in TEMPLATE_PATTERNS)),
            float(group_hits),
            float(sum(normalized.count(term) for term in INTENSIFIER_TERMS)),
            float(sum(normalized.count(term) for term in SARCASM_MARKERS)),
            float(sum(normalized.count(term) for term in CONTRAST_MARKERS)),
            float(sum(normalized.count(term) for term in MEME_PHRASES)),
            float(len(re.findall(r"[!?！？]{2,}", normalized))),
            float(len(re.findall(r"(.)\1{2,}", normalized))),
            1.0 if group_hits > 0 and risk_hits > 0 else 0.0,
            float(sum(normalized.count(term) for term in MITIGATION_TERMS)),
            float(sum(normalized.count(term) for term in REPORTING_TERMS)),
            float(sum(normalized.count(term) for term in ANALYSIS_TERMS)),
            float(sum(bool(pattern.search(normalized)) for pattern in STRONG_ATTACK_PATTERNS)),
            1.0
            if group_hits > 0 and sum(normalized.count(term) for term in MEME_PHRASES) > 0
            else 0.0,
        ]


def get_artifact_paths(artifact_dir: Path) -> ArtifactPaths:
    return ArtifactPaths(
        model_path=artifact_dir / "model.joblib",
        metrics_path=artifact_dir / "metrics.json",
        metadata_path=artifact_dir / "metadata.json",
    )


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"dataset not found: {data_path}")

    encodings = ("utf-8", "utf-8-sig", "gb18030", "gbk")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            dataset = pd.read_csv(data_path, encoding=encoding)
            break
        except Exception as error:  # pragma: no cover
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


def get_feature_summary(model_type: str) -> dict[str, Any]:
    if model_type == "baseline":
        return {
            "vectorizer": "char_tfidf_1_3",
            "extra_features": [],
        }

    return {
        "vectorizer": "char_tfidf_1_3",
        "extra_features": MemeExpressionFeatureExtractor.feature_names,
    }


def build_pipeline(model_type: str = DEFAULT_MODEL_TYPE) -> Pipeline:
    validate_model_type(model_type)

    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        min_df=2,
        lowercase=False,
        sublinear_tf=True,
    )

    if model_type == "baseline":
        feature_block: Any = tfidf
        feature_step_name = "tfidf"
    else:
        feature_block = FeatureUnion(
            transformer_list=[
                ("tfidf", tfidf),
                ("meme_features", MemeExpressionFeatureExtractor()),
            ]
        )
        feature_step_name = "features"

    return Pipeline(
        steps=[
            (feature_step_name, feature_block),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def compute_metrics(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
    }


def validate_model_type(model_type: str) -> None:
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"unsupported model_type: {model_type}. "
            f"expected one of {sorted(SUPPORTED_MODEL_TYPES)}"
        )


def train_and_save_model(
    data_path: Path,
    artifact_dir: Path,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> dict[str, Any]:
    validate_model_type(model_type)
    dataset = load_dataset(data_path)

    train_df, test_df = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"],
    )

    pipeline = build_pipeline(model_type=model_type)
    pipeline.fit(train_df["text"], train_df["label"])

    predictions = pipeline.predict(test_df["text"])
    metrics = compute_metrics(test_df["label"], predictions)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = get_artifact_paths(artifact_dir)

    joblib.dump(pipeline, artifact_paths.model_path)

    metadata = {
        "model_type": model_type,
        "sample_count": int(len(dataset)),
        "train_count": int(len(train_df)),
        "test_count": int(len(test_df)),
        "label_distribution": {
            str(label): int(count)
            for label, count in dataset["label"].value_counts().sort_index().items()
        },
        "data_path": str(data_path),
        "feature_summary": get_feature_summary(model_type),
    }

    artifact_paths.metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_paths.metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "artifact_dir": str(artifact_dir),
        "model_type": model_type,
        "model_path": str(artifact_paths.model_path),
        "metrics_path": str(artifact_paths.metrics_path),
        "metadata_path": str(artifact_paths.metadata_path),
        "sample_count": metadata["sample_count"],
        "train_count": metadata["train_count"],
        "test_count": metadata["test_count"],
        "label_distribution": metadata["label_distribution"],
        "metrics": metrics,
    }


class TextMemeClassifier:
    def __init__(
        self,
        data_path: Path,
        artifact_dir: Path | None = None,
        model_type: str = DEFAULT_MODEL_TYPE,
    ) -> None:
        validate_model_type(model_type)
        self.data_path = data_path
        self.artifact_dir = artifact_dir
        self.model_type = model_type
        self.mode = "rule"
        self.sample_count = 0
        self.metrics: dict[str, float] = {}
        self.model: Pipeline | None = None
        self.label_mapping = LABEL_MAPPING
        self.risk_keywords = RISK_KEYWORDS

    def initialize(self) -> None:
        if self._load_saved_model():
            self.mode = "artifact"
            return

        try:
            dataset = load_dataset(self.data_path)
            self.sample_count = len(dataset)
            self._train_model_in_memory(dataset)
            self.mode = "ml"
        except Exception:
            self.mode = "rule"
            self.model = None
            self.metrics = {}

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
        context = self._analyze_expression_context(clean_text)
        hit_count = context["risk_hits"]

        # 对明显攻击性表达增加一层增强；如果是反对/描述性语境，则避免被敏感词直接拉高。
        if context["direct_attack"] or context["meme_attack"]:
            score = max(model_score, min(0.6 + 0.06 * max(hit_count, 1), 0.9))
            label = 1
            message = "模型结果已结合模因式表达与攻击模式进行增强判断，建议进一步人工审核。"
        elif hit_count > 0 and context["mitigated"]:
            score = min(model_score, 0.45)
            label = 0
            message = "文本包含敏感词，但更接近描述、反对或分析语境，系统未直接判为风险文本。"
        elif hit_count > 0:
            score = max(model_score, min(0.48 + 0.04 * hit_count, 0.72))
            label = int(score >= 0.5)
            if label == 1:
                message = "模型根据文本内容与风险词信号判定该文本具有一定风险。"
            else:
                message = "文本包含部分风险信号，但当前证据不足以直接判为风险文本。"
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

    def _load_saved_model(self) -> bool:
        if self.artifact_dir is None:
            return False

        artifact_paths = get_artifact_paths(self.artifact_dir)
        if not artifact_paths.model_path.exists():
            return False

        self.model = joblib.load(artifact_paths.model_path)

        if artifact_paths.metrics_path.exists():
            self.metrics = json.loads(
                artifact_paths.metrics_path.read_text(encoding="utf-8")
            )

        if artifact_paths.metadata_path.exists():
            metadata = json.loads(
                artifact_paths.metadata_path.read_text(encoding="utf-8")
            )
            self.sample_count = int(metadata.get("sample_count", 0))
            self.model_type = metadata.get("model_type", self.model_type)

        return True

    def _train_model_in_memory(self, dataset: pd.DataFrame) -> None:
        train_df, test_df = train_test_split(
            dataset,
            test_size=0.2,
            random_state=42,
            stratify=dataset["label"],
        )

        pipeline = build_pipeline(model_type=self.model_type)
        pipeline.fit(train_df["text"], train_df["label"])

        predictions = pipeline.predict(test_df["text"])
        self.metrics = compute_metrics(test_df["label"], predictions)
        self.model = pipeline

    def _predict_with_rules(self, text: str) -> PredictionResult:
        context = self._analyze_expression_context(text)
        hit_count = context["risk_hits"]
        if context["direct_attack"] or context["meme_attack"]:
            label = 1
            score = min(0.6 + 0.06 * max(hit_count, 1), 0.9)
            message = "当前使用规则模式，文本触发了明显攻击性或模因式风险表达。"
        elif hit_count > 0 and context["mitigated"]:
            label = 0
            score = 0.3
            message = "当前使用规则模式，文本包含敏感词，但整体更接近描述或反对语境。"
        elif hit_count > 0:
            label = 1 if hit_count >= 2 else 0
            score = min(0.46 + 0.05 * hit_count, 0.75) if label == 1 else 0.4
            message = (
                "当前使用规则模式，文本存在一定风险信号。"
                if label == 1
                else "当前使用规则模式，文本存在弱风险信号，但证据不足。"
            )
        else:
            label = 0
            score = 0.18
            message = "当前使用规则模式，文本未触发明显风险关键词。"

        return PredictionResult(
            label=label,
            label_name=self.label_mapping[label],
            score=round(float(score), 4),
            message=message,
        )

    def _count_risk_keywords(self, text: str) -> int:
        return sum(1 for keyword in self.risk_keywords if keyword in text)

    def _analyze_expression_context(self, text: str) -> dict[str, Any]:
        group_hits = sum(text.count(term) for term in GROUP_TERMS)
        risk_hits = sum(text.count(term) for term in self.risk_keywords)
        mitigation_hits = sum(text.count(term) for term in MITIGATION_TERMS)
        reporting_hits = sum(text.count(term) for term in REPORTING_TERMS)
        analysis_hits = sum(text.count(term) for term in ANALYSIS_TERMS)
        meme_hits = sum(text.count(term) for term in MEME_PHRASES)
        direct_attack_pattern_hits = sum(
            bool(pattern.search(text)) for pattern in STRONG_ATTACK_PATTERNS
        )

        direct_attack = direct_attack_pattern_hits > 0 or (group_hits > 0 and risk_hits > 0)
        mitigated = (mitigation_hits + reporting_hits + analysis_hits) > 0 and not direct_attack
        meme_attack = meme_hits > 0 and (group_hits > 0 or risk_hits > 0) and not mitigated

        return {
            "group_hits": group_hits,
            "risk_hits": risk_hits,
            "mitigation_hits": mitigation_hits,
            "reporting_hits": reporting_hits,
            "analysis_hits": analysis_hits,
            "meme_hits": meme_hits,
            "direct_attack_pattern_hits": direct_attack_pattern_hits,
            "direct_attack": direct_attack,
            "mitigated": mitigated,
            "meme_attack": meme_attack,
        }
