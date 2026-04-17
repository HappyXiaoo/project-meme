# Back End

## 启动方式

1. 进入目录：`cd back-end`
2. 安装依赖：`pip install -r requirements.txt`
3. 先训练并保存模型：`python scripts/train_model.py`
4. 启动服务：`python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`

## 已实现接口

- `GET /health`
- `GET /labels`
- `POST /predict`

## 当前模型策略

- 优先读取项目根目录中的 `data/data.csv`
- 训练脚本会将模型保存到 `back-end/artifacts/`
- 使用中文字符级 `TF-IDF + LogisticRegression`
- 当前默认训练配置为 `meme_features_v1`
- 服务启动时优先加载已训练模型
- 如果本地模型不存在，则回退到内存训练；再失败时回退到规则判断模式

## 模型训练

执行：

```text
python scripts/train_model.py
```

如需训练基线模型，可执行：

```text
python scripts/train_model.py --model-type baseline
```

## BERT 版骨架

当前已经提供 BERT 版训练脚本和目录骨架：

- 训练入口：`python scripts/train_bert.py`
- 配置对象：`back-end/bert/config.py`
- 数据拆分：`back-end/bert/data.py`
- 训练流程：`back-end/bert/train.py`
- 依赖清单：`back-end/requirements-bert.txt`

建议训练方式：

1. 先在本地用 CPU 做 smoke test
2. 确认流程跑通后，再在 GPU 环境做正式训练

本地 smoke test 示例：

```text
python scripts/train_bert.py --smoke-test --cpu-only --train-sample-limit 128 --eval-sample-limit 64
```

正式训练示例：

```text
python scripts/train_bert.py --model-name bert-base-chinese --epochs 2
```

默认会输出到：

- `back-end/artifacts-bert-v1/model/`
- `back-end/artifacts-bert-v1/metrics.json`
- `back-end/artifacts-bert-v1/metadata.json`

注意事项：

- 当前机器上尚未安装 `torch` 和 `transformers`
- 如果本机 Python 版本无法安装 torch，建议在租用的 GPU 环境中创建 Python 3.10/3.11 虚拟环境后再训练
- BERT 版目前只完成了训练骨架，后续可再接入后端推理服务

默认会输出：

- `back-end/artifacts/model.joblib`
- `back-end/artifacts/metrics.json`
- `back-end/artifacts/metadata.json`

## 接口示例

请求：

```json
{
  "text": "这个群体根本不配得到尊重"
}
```

返回：

```json
{
  "label": 1,
  "label_name": "疑似有害文本",
  "score": 0.84,
  "message": "模型结果已结合风险关键词进行增强判断，建议进一步人工审核。"
}
```
