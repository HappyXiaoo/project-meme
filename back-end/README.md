# Back End

## 启动方式

1. 进入目录：`cd back-end`
2. 安装依赖：`pip install -r requirements.txt`
3. 启动服务：`python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`

## 已实现接口

- `GET /health`
- `GET /labels`
- `POST /predict`

## 当前模型策略

- 优先读取项目根目录中的 `data/data.csv`
- 使用中文字符级 `TF-IDF + LogisticRegression`
- 如果数据训练失败，则回退到规则判断模式

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
  "message": "模型已完成预测，可结合人工判断进一步审核。"
}
```
