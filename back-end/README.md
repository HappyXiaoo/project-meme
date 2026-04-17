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
- 服务启动时优先加载已训练模型
- 如果本地模型不存在，则回退到内存训练；再失败时回退到规则判断模式

## 模型训练

执行：

```text
python scripts/train_model.py
```

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
