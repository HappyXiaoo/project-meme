from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DEFAULT_SAMPLES = {
    "B01": "这个群体根本不配得到尊重。",
    "B02": "我反对任何形式的歧视言论。",
    "B03": "这种说法像是在侮辱别人。",
    "B04": "不是我说，某些人真的是典中典。",
    "B05": "果然某些人又开始这一套了，笑死。",
}

LABEL_MAPPING = {
    0: "正常文本",
    1: "疑似有害文本",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量调用已训练 BERT 模型预测样例文本。")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=BACKEND_DIR / "artifacts-bert-v1" / "model",
        help="BERT 模型目录，目录中应包含模型和 tokenizer。",
    )
    parser.add_argument(
        "--samples-json",
        type=Path,
        default=None,
        help="可选，自定义样例 JSON 文件路径。格式为 {\"编号\": \"文本\"}。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="预测时最大 token 长度。",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "markdown", "both"],
        default="both",
        help="输出格式。",
    )
    return parser.parse_args()


def load_samples(samples_json: Path | None) -> dict[str, str]:
    if samples_json is None:
        return DEFAULT_SAMPLES

    payload = json.loads(samples_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("samples json must be an object like {\"B01\": \"text\"}")

    normalized = {}
    for key, value in payload.items():
        normalized[str(key)] = str(value)
    return normalized


def predict_samples(model_dir: Path, samples: dict[str, str], max_length: int) -> list[dict[str, object]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as error:  # pragma: no cover
        raise RuntimeError("缺少 BERT 预测依赖，请先安装 torch 和 transformers。") from error

    if not model_dir.exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results: list[dict[str, object]] = []
    for sample_id, text in samples.items():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            label = int(torch.argmax(probabilities).item())
            score = float(probabilities[label].item())

        results.append(
            {
                "id": sample_id,
                "text": text,
                "label": label,
                "label_name": LABEL_MAPPING[label],
                "score": round(score, 4),
            }
        )

    return results


def to_markdown_table(results: list[dict[str, object]]) -> str:
    lines = [
        "| 编号 | 输入文本 | BERT 模型 |",
        "| --- | --- | --- |",
    ]
    for item in results:
        cell = f"{item['label_name']}（{item['score']:.4f}）"
        lines.append(f"| {item['id']} | {item['text']} | {cell} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    samples = load_samples(args.samples_json)
    results = predict_samples(
        model_dir=args.model_dir.resolve(),
        samples=samples,
        max_length=args.max_length,
    )

    if args.output_format in {"json", "both"}:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output_format in {"markdown", "both"}:
        if args.output_format == "both":
            print()
        print(to_markdown_table(results))


if __name__ == "__main__":
    main()
