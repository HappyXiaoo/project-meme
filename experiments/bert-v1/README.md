# BERT V1

当前目录用于保存 `BERT` 初版实验材料。

## 当前骨架

- 训练入口：`back-end/scripts/train_bert.py`
- 配置文件：`back-end/bert/config.py`
- 数据处理：`back-end/bert/data.py`
- 训练流程：`back-end/bert/train.py`
- 依赖清单：`back-end/requirements-bert.txt`

## 计划记录内容

- 模型结构说明
- 训练参数
- CPU smoke test 结果
- GPU 正式训练结果
- 与基线模型、模因特征模型的对比结果

## 当前已完成结果

- GPU：RTX 3090 24GB
- 模型：`bert-base-chinese`
- 正式训练参数：
  - epochs: 2
  - max_length: 128
  - train_batch_size: 8
  - eval_batch_size: 8
  - gradient_accumulation_steps: 1
- 数据规模：
  - train_count: 10776
  - eval_count: 1347
  - test_count: 1348
- 指标：
  - accuracy: 0.8887
  - precision: 0.8515
  - recall: 0.8851
  - f1: 0.8680

## 当前结论

- BERT 正式训练结果优于基线模型
- BERT 正式训练结果优于当前 `meme_features_v1`
- 后续需要补充典型样例对比，完善论文中的案例分析部分
