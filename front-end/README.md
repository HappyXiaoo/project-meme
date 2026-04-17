# Front End

## 启动方式

1. 进入目录：`cd front-end`
2. 安装依赖：`npm install`
3. 启动开发环境：`npm run dev`

## 当前完成内容

- 首页视觉布局
- 文本输入区
- 检测结果区
- 历史记录区
- 前端 mock 预测流程
- 后端 `POST /predict` 接口预留

## 当前接口约定

前端会优先请求：

```text
POST http://127.0.0.1:8000/predict
Content-Type: application/json
```

请求体：

```json
{
  "text": "待检测文本"
}
```

期望返回：

```json
{
  "label": 1,
  "label_name": "疑似有害文本",
  "score": 0.87,
  "message": "系统判断该文本存在较高风险"
}
```

如果后端尚未启动，前端会自动回退到本地 mock 数据，方便页面先行开发。
