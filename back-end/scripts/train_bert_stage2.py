from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用模因式表达增强数据对已训练 BERT 模型进行二阶段微调。")
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=BACKEND_DIR / "artifacts-bert-v1" / "model",
        help="第一阶段已训练好的 BERT 模型目录。",
    )
    parser.add_argument(
        "--augmented-data-path",
        type=Path,
        default=PROJECT_DIR / "data" / "meme_augmented_train_set.csv",
        help="模因式表达增强训练集路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BACKEND_DIR / "artifacts-bert-stage2-v1",
        help="二阶段微调模型输出目录。",
    )
    parser.add_argument("--max-length", type=int, default=128, help="最大 token 长度。")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="学习率。")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数。")
    parser.add_argument("--train-batch-size", type=int, default=4, help="训练 batch size。")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="验证和测试 batch size。")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="梯度累积步数。")
    parser.add_argument("--cpu-only", action="store_true", help="强制使用 CPU。")
    parser.add_argument("--smoke-test", action="store_true", help="仅用少量样本验证流程。")
    parser.add_argument("--train-sample-limit", type=int, default=None, help="限制训练样本数。")
    parser.add_argument("--eval-sample-limit", type=int, default=None, help="限制验证和测试样本数。")
    return parser.parse_args()


def main() -> None:
    try:
        import pandas as pd
        import torch
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except Exception as error:  # pragma: no cover
        raise RuntimeError("缺少二阶段微调所需依赖，请先安装 pandas、torch、transformers、sklearn。") from error

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(42)

    df = pd.read_csv(args.augmented_data_path, encoding="utf-8")
    df = df.loc[:, ["label", "text"]].copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    eval_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"],
    )

    if args.smoke_test:
        args.train_sample_limit = args.train_sample_limit or 32
        args.eval_sample_limit = args.eval_sample_limit or 16

    if args.train_sample_limit and len(train_df) > args.train_sample_limit:
        train_df = train_df.sample(n=args.train_sample_limit, random_state=42)
    if args.eval_sample_limit and len(eval_df) > args.eval_sample_limit:
        eval_df = eval_df.sample(n=args.eval_sample_limit, random_state=42)
        test_df = test_df.sample(n=min(args.eval_sample_limit, len(test_df)), random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model_dir)

    class Stage2Dataset:
        def __init__(self, frame):
            self.texts = frame["text"].tolist()
            self.labels = frame["label"].tolist()

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoded = tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=args.max_length,
                padding=False,
            )
            encoded["labels"] = self.labels[idx]
            return encoded

    train_dataset = Stage2Dataset(train_df.reset_index(drop=True))
    eval_dataset = Stage2Dataset(eval_df.reset_index(drop=True))
    test_dataset = Stage2Dataset(test_df.reset_index(drop=True))

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.cpu_only:
        fp16 = False
        no_cuda = True
        device_name = "cpu"
    else:
        has_cuda = torch.cuda.is_available()
        fp16 = has_cuda
        no_cuda = not has_cuda
        device_name = "cuda" if has_cuda else "cpu"

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        fp16=fp16,
        no_cuda=no_cuda,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": round(float(accuracy_score(labels, preds)), 4),
            "precision": round(float(precision_score(labels, preds)), 4),
            "recall": round(float(recall_score(labels, preds)), 4),
            "f1": round(float(f1_score(labels, preds)), 4),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    final_metrics = {
        "accuracy": round(float(test_metrics["test_accuracy"]), 4),
        "precision": round(float(test_metrics["test_precision"]), 4),
        "recall": round(float(test_metrics["test_recall"]), 4),
        "f1": round(float(test_metrics["test_f1"]), 4),
    }

    save_dir = args.output_dir / "model"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    metadata = {
        "model_type": "bert_stage2_v1",
        "base_model_dir": str(args.base_model_dir),
        "augmented_data_path": str(args.augmented_data_path),
        "device": device_name,
        "smoke_test": args.smoke_test,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_count": len(train_df),
        "eval_count": len(eval_df),
        "test_count": len(test_df),
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "model_dir": str(save_dir),
                "metrics": final_metrics,
                "metadata": metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
