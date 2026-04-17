from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.classifier import DEFAULT_MODEL_TYPE, SUPPORTED_MODEL_TYPES, train_and_save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练文本模因识别模型并保存产物。")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_DIR / "data" / "data.csv",
        help="训练数据 CSV 文件路径。",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=BACKEND_DIR / "artifacts",
        help="模型与指标输出目录。",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=sorted(SUPPORTED_MODEL_TYPES),
        help="模型类型，可选 baseline 或 meme_features_v1。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_and_save_model(
        data_path=args.data_path.resolve(),
        artifact_dir=args.artifact_dir.resolve(),
        model_type=args.model_type,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
