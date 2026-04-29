from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from bert_fusion import BertMemeFusionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 BERT+模因特征融合模型并保存产物。")
    parser.add_argument("--data-path", type=Path, default=PROJECT_DIR / "data" / "data.csv", help="训练数据 CSV 文件路径。")
    parser.add_argument("--output-dir", type=Path, default=BACKEND_DIR / "artifacts-bert-meme-v1", help="融合模型产物输出目录。")
    parser.add_argument("--model-name", type=str, default="bert-base-chinese", help="Hugging Face 预训练模型名称。")
    parser.add_argument("--max-length", type=int, default=128, help="最大 token 长度。")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="学习率。")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数。")
    parser.add_argument("--train-batch-size", type=int, default=4, help="训练 batch size。")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="验证和测试 batch size。")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="梯度累积步数。")
    parser.add_argument("--fusion-hidden-size", type=int, default=128, help="融合层隐藏维度。")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout 比例。")
    parser.add_argument("--smoke-test", action="store_true", help="仅用少量样本验证训练流程。")
    parser.add_argument("--cpu-only", action="store_true", help="强制使用 CPU。")
    parser.add_argument("--train-sample-limit", type=int, default=None, help="限制训练样本数。")
    parser.add_argument("--eval-sample-limit", type=int, default=None, help="限制验证和测试样本数。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from bert_fusion.train import train_bert_meme_fusion_pipeline

    config = BertMemeFusionConfig(
        data_path=args.data_path.resolve(),
        output_dir=args.output_dir.resolve(),
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fusion_hidden_size=args.fusion_hidden_size,
        dropout=args.dropout,
        smoke_test=args.smoke_test,
        cpu_only=args.cpu_only,
        train_sample_limit=args.train_sample_limit,
        eval_sample_limit=args.eval_sample_limit,
    )
    summary = train_bert_meme_fusion_pipeline(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
