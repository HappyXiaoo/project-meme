from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .config import BertTrainingConfig
from .data import BertTextDataset, build_data_splits


def train_bert_pipeline(config: BertTrainingConfig) -> dict[str, object]:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except Exception as error:  # pragma: no cover
        raise RuntimeError(
            "BERT 训练依赖未安装。请先安装 torch、transformers、accelerate。"
        ) from error

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.cpu_only:
        use_fp16 = False
        no_cuda = True
        device_name = "cpu"
    else:
        has_cuda = torch.cuda.is_available()
        use_fp16 = has_cuda
        no_cuda = not has_cuda
        device_name = "cuda" if has_cuda else "cpu"

    set_seed(config.random_seed)

    splits = build_data_splits(
        data_path=config.data_path,
        random_seed=config.random_seed,
        smoke_test=config.smoke_test,
        train_sample_limit=config.train_sample_limit,
        eval_sample_limit=config.eval_sample_limit,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    )

    train_dataset = BertTextDataset(splits.train_df, tokenizer, config.max_length)
    eval_dataset = BertTextDataset(splits.eval_df, tokenizer, config.max_length)
    test_dataset = BertTextDataset(splits.test_df, tokenizer, config.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=config.save_total_limit,
        fp16=use_fp16,
        no_cuda=no_cuda,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return {
            "accuracy": round(float(accuracy_score(labels, predictions)), 4),
            "precision": round(float(precision_score(labels, predictions)), 4),
            "recall": round(float(recall_score(labels, predictions)), 4),
            "f1": round(float(f1_score(labels, predictions)), 4),
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

    save_dir = config.output_dir / "model"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    metadata = {
        "model_type": "bert_v1",
        "model_name": config.model_name,
        "device": device_name,
        "smoke_test": config.smoke_test,
        "cpu_only": config.cpu_only,
        "max_length": config.max_length,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "train_count": len(splits.train_df),
        "eval_count": len(splits.eval_df),
        "test_count": len(splits.test_df),
        "data_path": str(config.data_path),
    }

    (config.output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": str(config.output_dir),
        "model_dir": str(save_dir),
        "metrics": final_metrics,
        "metadata": metadata,
    }
