# GPU 环境部署与 BERT 训练步骤

这份文档用于在租用 GPU 后，快速完成 `BERT` 版环境搭建、训练、结果保存和论文材料整理。

## 1. 推荐环境

建议优先选择以下环境：

- 操作系统：`Ubuntu 20.04` 或 `Ubuntu 22.04`
- Python：`3.10` 或 `3.11`
- GPU：显存 `8GB` 及以上
- CUDA：优先选择平台已预装、且 `torch` 可直接使用的环境

建议原则：

- 优先选已经配置好 `CUDA + Python + pip` 的镜像
- 不要在 `Python 3.14` 环境里硬装深度学习依赖
- 优先保证“能跑通”而不是“环境最先进”

## 2. 训练前准备

建议提前准备：

- 项目代码仓库
- 数据文件 `data/data.csv`
- 当前实验记录文档
- 本地已完成的基线模型与模因特征模型指标

建议在租用机器后先确认：

- 是否能正常联网
- 是否有可写磁盘空间
- 是否能使用 GPU

可先执行：

```bash
python --version
nvidia-smi
```

如果 `nvidia-smi` 能正常显示 GPU 信息，说明 GPU 环境基本正常。

## 3. 代码上传与目录确认

将项目同步到 GPU 机器后，确认目录结构至少包含：

```text
project-meme/
  back-end/
  data/
  docs/
```

重点确认以下文件存在：

- `back-end/scripts/train_bert.py`
- `back-end/bert/config.py`
- `back-end/bert/data.py`
- `back-end/bert/train.py`
- `back-end/requirements-bert.txt`
- `data/data.csv`

## 4. 创建虚拟环境

建议在 GPU 机器上单独创建虚拟环境：

```bash
cd project-meme/back-end
python -m venv .venv
source .venv/bin/activate
```

更新基础工具：

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 5. 安装依赖

先安装后端基础依赖：

```bash
pip install -r requirements.txt
```

再安装 BERT 相关依赖：

```bash
pip install -r requirements-bert.txt
```

如果 `torch` 安装失败：

- 优先检查 Python 版本是否为 `3.10/3.11`
- 优先使用租用平台自带的 `torch` 环境
- 必要时根据 CUDA 版本单独安装 `torch`

## 6. 先做一次最小验证

在正式训练前，建议先执行：

```bash
python scripts/train_bert.py --smoke-test --train-sample-limit 128 --eval-sample-limit 64
```

这一步的目标不是拿正式结果，而是确认：

- tokenizer 能正常加载
- 模型能正常加载
- 数据读取没问题
- 训练脚本能正常运行
- 能输出模型目录、`metrics.json`、`metadata.json`

如果 smoke test 成功，再进行正式训练。

## 7. 正式训练命令

建议第一版正式训练先用保守参数：

```bash
python scripts/train_bert.py \
  --model-name bert-base-chinese \
  --epochs 2 \
  --max-length 128 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --gradient-accumulation-steps 1
```

如果显存不足，可以这样调整：

- 将 `--train-batch-size` 改成 `4`
- 仍然不够时，将 `--max-length` 改成 `96`
- 再不够时，增加 `--gradient-accumulation-steps`

例如：

```bash
python scripts/train_bert.py \
  --model-name bert-base-chinese \
  --epochs 2 \
  --max-length 96 \
  --train-batch-size 4 \
  --eval-batch-size 8 \
  --gradient-accumulation-steps 2
```

## 8. 训练完成后检查的文件

正常情况下，训练完成后应生成：

```text
back-end/artifacts-bert-v1/
  model/
  metrics.json
  metadata.json
  checkpoints/
```

重点检查：

- `model/` 是否存在
- `metrics.json` 是否存在
- `metadata.json` 是否存在

## 9. 训练结果需要记录什么

训练完成后，建议立刻把以下信息抄回论文材料：

- 训练日期
- GPU 型号
- 显存大小
- 模型名称
- 学习率
- epoch 数
- batch size
- 最大长度
- accuracy
- precision
- recall
- f1
- 总训练时长

这些内容建议补到：

- [实验记录.md](/d:/code/project-meme/docs/实验记录.md)
- [测试样例.md](/d:/code/project-meme/docs/测试样例.md)

## 10. 三模型对比建议

正式训练完成后，建议至少做以下对比：

- 基线模型
- `meme_features_v1`
- `BERT`

建议对比两类内容：

### 10.1 指标对比

- accuracy
- precision
- recall
- f1

### 10.2 样例对比

重点填写这些样例：

- 明显正常文本
- 明显风险文本
- 容易误判文本
- 模因式表达文本

## 11. 常见问题处理

### 11.1 显存不足

处理方式：

- 降低 `train_batch_size`
- 降低 `max_length`
- 增加 `gradient_accumulation_steps`

### 11.2 下载模型慢

处理方式：

- 提前确认网络是否稳定
- 如果平台支持，优先使用镜像源或预缓存模型环境

### 11.3 训练太慢

处理方式：

- 先做 `smoke test`
- 正式训练时只先跑 `2` 个 epoch
- 不要一开始追求过深调参

### 11.4 指标不如传统模型

处理方式：

- 这不代表实验失败
- 依然可以写进论文
- 重点分析典型样例和语义理解能力差异

## 12. 建议的训练后收尾动作

训练完成后，建议按顺序做：

1. 保存训练日志
2. 保存 `metrics.json` 和 `metadata.json`
3. 记录关键训练参数
4. 用几条测试样例做预测
5. 回填实验记录
6. 回填测试样例对比表
7. 整理为论文第 4 章材料

## 13. 一句话执行顺序

拿到 GPU 后，按这个顺序做就够了：

1. 建虚拟环境
2. 装依赖
3. 跑 smoke test
4. 跑正式训练
5. 保存结果
6. 回填文档
