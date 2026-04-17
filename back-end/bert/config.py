from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BertTrainingConfig:
    data_path: Path
    output_dir: Path
    model_name: str = "bert-base-chinese"
    max_length: int = 128
    learning_rate: float = 2e-5
    num_train_epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    random_seed: int = 42
    smoke_test: bool = False
    cpu_only: bool = False
    train_sample_limit: int | None = None
    eval_sample_limit: int | None = None
    logging_steps: int = 20
    save_total_limit: int = 2
