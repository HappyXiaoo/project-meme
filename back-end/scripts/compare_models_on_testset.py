from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.classifier import LABEL_MAPPING, TextMemeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在人工测试集上对比基线模型、模因特征模型、BERT 模型、融合模型和二阶段微调模型。")
    parser.add_argument(
        "--testset",
        type=Path,
        default=PROJECT_DIR / "data" / "manual_test_set.csv",
        help="人工测试集 CSV 文件路径。",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_DIR / "data" / "data.csv",
        help="训练数据 CSV 文件路径，用于传统模型回退训练。",
    )
    parser.add_argument(
        "--baseline-artifact-dir",
        type=Path,
        default=BACKEND_DIR / "artifacts-baseline",
        help="基线模型产物目录。如果不存在，将根据当前代码临时训练。",
    )
    parser.add_argument(
        "--meme-artifact-dir",
        type=Path,
        default=BACKEND_DIR / "artifacts",
        help="模因特征模型产物目录。如果不存在，将根据当前代码临时训练。",
    )
    parser.add_argument(
        "--bert-model-dir",
        type=Path,
        default=BACKEND_DIR / "artifacts-bert-v1" / "model",
        help="BERT 模型目录。",
    )
    parser.add_argument(
        "--bert-meme-model-dir",
        type=Path,
        default=BACKEND_DIR / "artifacts-bert-meme-v1" / "model",
        help="BERT+模因特征融合模型目录。",
    )
    parser.add_argument(
        "--bert-stage2-model-dir",
        type=Path,
        default=BACKEND_DIR / "artifacts-bert-stage2-v1" / "model",
        help="BERT 二阶段微调模型目录。",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="可选，保存详细对比结果 CSV。",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "markdown", "both"],
        default="both",
        help="输出格式。",
    )
    return parser.parse_args()


def load_testset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    required = {"id", "label", "category", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"testset must contain columns: {sorted(required)}")
    df = df.loc[:, ["id", "label", "category", "text", "expected_note"]].copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)
    return df


def run_traditional_model(
    data_path: Path,
    artifact_dir: Path,
    model_type: str,
    texts: list[str],
) -> tuple[str, list[dict[str, object]]]:
    classifier = TextMemeClassifier(
        data_path=data_path,
        artifact_dir=artifact_dir,
        model_type=model_type,
    )
    classifier.initialize()
    predictions = []
    for text in texts:
        result = classifier.predict(text)
        predictions.append(
            {
                "label": result.label,
                "label_name": result.label_name,
                "score": round(float(result.score), 4),
            }
        )
    return classifier.mode, predictions


def run_bert_model(model_dir: Path, texts: list[str]) -> tuple[str, list[dict[str, object]]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as error:  # pragma: no cover
        raise RuntimeError("缺少 BERT 预测依赖，请先安装 torch 和 transformers。") from error

    if not model_dir.exists():
        raise FileNotFoundError(f"bert model dir not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    predictions = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            label = int(torch.argmax(probabilities).item())
            score = float(probabilities[label].item())
        predictions.append(
            {
                "label": label,
                "label_name": LABEL_MAPPING[label],
                "score": round(score, 4),
            }
        )

    return device, predictions


def run_bert_stage2_model(model_dir: Path, texts: list[str]) -> tuple[str, list[dict[str, object]]]:
    # 二阶段微调后的模型仍然是标准 BERT 分类模型，直接复用原始 BERT 加载逻辑。
    return run_bert_model(model_dir, texts)


def run_bert_meme_model(model_dir: Path, texts: list[str]) -> tuple[str, list[dict[str, object]]]:
    try:
        import torch
        from transformers import AutoTokenizer

        from app.services.classifier import extract_meme_feature_vector
        from bert_fusion.model import BertMemeFusionModel
    except Exception as error:  # pragma: no cover
        raise RuntimeError("缺少融合模型预测依赖，请先安装 torch 和 transformers。") from error

    if not model_dir.exists():
        raise FileNotFoundError(f"bert meme model dir not found: {model_dir}")

    metadata_path = model_dir.parent / "metadata.json"
    fusion_hidden_size = 128
    dropout = 0.1
    base_model_name = "bert-base-chinese"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fusion_hidden_size = int(metadata.get("fusion_hidden_size", fusion_hidden_size))
        dropout = float(metadata.get("dropout", dropout))
        base_model_name = metadata.get("model_name", base_model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = BertMemeFusionModel.from_pretrained_fusion(
        model_dir,
        model_name=base_model_name,
        fusion_hidden_size=fusion_hidden_size,
        dropout=dropout,
        num_labels=2,
        torch_module=torch,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    predictions = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        meme_features = torch.tensor([extract_meme_feature_vector(text)], dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = model(**inputs, meme_features=meme_features)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            label = int(torch.argmax(probabilities).item())
            score = float(probabilities[label].item())
        predictions.append(
            {
                "label": label,
                "label_name": LABEL_MAPPING[label],
                "score": round(score, 4),
            }
        )

    return device, predictions


def add_prediction_columns(
    df: pd.DataFrame,
    prefix: str,
    predictions: list[dict[str, object]],
) -> None:
    df[f"{prefix}_label"] = [item["label"] for item in predictions]
    df[f"{prefix}_label_name"] = [item["label_name"] for item in predictions]
    df[f"{prefix}_score"] = [item["score"] for item in predictions]
    df[f"{prefix}_correct"] = (df["label"] == df[f"{prefix}_label"]).astype(int)


def build_summary(df: pd.DataFrame) -> dict[str, object]:
    model_prefixes = ["baseline", "meme", "bert", "bert_meme", "bert_stage2"]
    overall = {}
    by_category = {}

    for prefix in model_prefixes:
        overall[prefix] = {
            "accuracy": round(float(df[f"{prefix}_correct"].mean()), 4),
        }

    for category, group in df.groupby("category"):
        by_category[category] = {}
        for prefix in model_prefixes:
            by_category[category][prefix] = {
                "accuracy": round(float(group[f"{prefix}_correct"].mean()), 4),
                "count": int(len(group)),
            }

    return {
        "overall_accuracy": overall,
        "by_category": by_category,
    }


def format_prediction_cell(label_name: str, score: float) -> str:
    return f"{label_name}（{score:.4f}）"


def to_markdown(df: pd.DataFrame, summary: dict[str, object]) -> str:
    lines = [
        "## 五模型对比结果",
        "",
        "| 编号 | 类别 | 预期结果 | 基线模型 | 模因特征模型 | BERT 模型 | 融合模型 | 二阶段微调 BERT |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| {id} | {category} | {expected} | {baseline} | {meme} | {bert} | {bert_meme} | {bert_stage2} |".format(
                id=row["id"],
                category=row["category"],
                expected=LABEL_MAPPING[int(row["label"])],
                baseline=format_prediction_cell(row["baseline_label_name"], row["baseline_score"]),
                meme=format_prediction_cell(row["meme_label_name"], row["meme_score"]),
                bert=format_prediction_cell(row["bert_label_name"], row["bert_score"]),
                bert_meme=format_prediction_cell(row["bert_meme_label_name"], row["bert_meme_score"]),
                bert_stage2=format_prediction_cell(row["bert_stage2_label_name"], row["bert_stage2_score"]),
            )
        )

    lines.extend(
        [
            "",
            "## 总体准确率",
            "",
            "| 模型 | 准确率 |",
            "| --- | --- |",
            f"| baseline | {summary['overall_accuracy']['baseline']['accuracy']:.4f} |",
            f"| meme_features_v1 | {summary['overall_accuracy']['meme']['accuracy']:.4f} |",
            f"| bert | {summary['overall_accuracy']['bert']['accuracy']:.4f} |",
            f"| bert_meme_fusion | {summary['overall_accuracy']['bert_meme']['accuracy']:.4f} |",
            f"| bert_stage2 | {summary['overall_accuracy']['bert_stage2']['accuracy']:.4f} |",
            "",
            "## 分类别准确率",
            "",
            "| 类别 | 样本数 | baseline | meme_features_v1 | bert | bert_meme_fusion | bert_stage2 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for category, values in summary["by_category"].items():
        lines.append(
            f"| {category} | {values['baseline']['count']} | "
            f"{values['baseline']['accuracy']:.4f} | "
            f"{values['meme']['accuracy']:.4f} | "
            f"{values['bert']['accuracy']:.4f} | "
            f"{values['bert_meme']['accuracy']:.4f} | "
            f"{values['bert_stage2']['accuracy']:.4f} |"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    df = load_testset(args.testset.resolve())
    texts = df["text"].tolist()

    baseline_mode, baseline_predictions = run_traditional_model(
        data_path=args.data_path.resolve(),
        artifact_dir=args.baseline_artifact_dir.resolve(),
        model_type="baseline",
        texts=texts,
    )
    meme_mode, meme_predictions = run_traditional_model(
        data_path=args.data_path.resolve(),
        artifact_dir=args.meme_artifact_dir.resolve(),
        model_type="meme_features_v1",
        texts=texts,
    )
    bert_device, bert_predictions = run_bert_model(
        model_dir=args.bert_model_dir.resolve(),
        texts=texts,
    )
    bert_meme_device, bert_meme_predictions = run_bert_meme_model(
        model_dir=args.bert_meme_model_dir.resolve(),
        texts=texts,
    )
    bert_stage2_device, bert_stage2_predictions = run_bert_stage2_model(
        model_dir=args.bert_stage2_model_dir.resolve(),
        texts=texts,
    )

    add_prediction_columns(df, "baseline", baseline_predictions)
    add_prediction_columns(df, "meme", meme_predictions)
    add_prediction_columns(df, "bert", bert_predictions)
    add_prediction_columns(df, "bert_meme", bert_meme_predictions)
    add_prediction_columns(df, "bert_stage2", bert_stage2_predictions)

    summary = build_summary(df)
    payload = {
        "metadata": {
            "baseline_mode": baseline_mode,
            "meme_mode": meme_mode,
            "bert_device": bert_device,
            "bert_meme_device": bert_meme_device,
            "bert_stage2_device": bert_stage2_device,
            "test_count": int(len(df)),
        },
        "summary": summary,
        "results": df.to_dict(orient="records"),
    }

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    if args.output_format in {"json", "both"}:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output_format in {"markdown", "both"}:
        if args.output_format == "both":
            print()
        print(to_markdown(df, summary))


if __name__ == "__main__":
    main()
