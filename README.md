# MGPT: Multi-Grained Preference Enhanced Transformer

这是论文 **"Multi-Grained Preference Enhanced Transformer for Multi-behavior Sequential Recommendation"** 的官方实现代码。

## 简介

MGPT 是一个针对多行为序列推荐的深度学习模型，通过多粒度偏好增强的 Transformer 架构来建模用户的多种交互行为（如浏览、收藏、购买等），从而提升推荐性能。

本项目基于 [RecBole](https://github.com/RUCAIBox/RecBole) 框架实现。

## 环境要求

- Python >= 3.7
- PyTorch >= 1.10.0
- CUDA >= 10.2 (推荐使用 GPU)

## 安装依赖

```bash
pip install torch torchvision
pip install numpy scipy pandas
pip install scikit-learn
pip install tqdm
pip install pyyaml
```

或者使用 RecBole 框架：

```bash
pip install recbole
```

## 数据集准备

本项目支持以下多行为推荐数据集：

- **Tmall** (`tmall_beh`)
- **IJCAI** (`ijcai_beh`)
- **Retail Rocket** (`retail_beh`)

### 数据格式

数据集应放置在 `dataset/数据集名称/` 目录下，包含以下文件：

- `数据集名称.inter`：交互数据文件

数据文件格式应包含以下字段：
- `session_id`：用户会话 ID
- `item_id_list`：物品 ID 序列
- `item_type_list`：行为类型序列（0表示购买，其他数字表示不同类型的辅助行为）
- `item_id`：目标物品 ID

示例：
```
session_id:token    item_id_list:token_seq    item_type_list:token_seq    item_id:token
1    1 2 3    1 2 0    4
2    5 6 7 8    2 1 1 0    9
```

## 快速开始

### 训练模型

在 **Tmall** 数据集上训练：

```bash
python run_M-GPT.py --model MGPT --dataset tmall_beh --gpu_id 0
```

在 **IJCAI** 数据集上训练：

```bash
python run_M-GPT.py --model MGPT --dataset ijcai_beh --gpu_id 0
```

在 **Retail Rocket** 数据集上训练：

```bash
python run_M-GPT.py --model MGPT --dataset retail_beh --gpu_id 0
```

### 使用验证集

如果想使用验证集进行调参：

```bash
python run_M-GPT.py --model MGPT --dataset tmall_beh --validation --valid_portion 0.1 --gpu_id 0
```

### 自定义批大小

```bash
python run_M-GPT.py --model MGPT --dataset tmall_beh --batch_size 64 --gpu_id 0
```

## 模型参数

主要的超参数配置在 `run_M-GPT.py` 中的 `config_dict` 中：

### 核心参数

- `n_layers`：Transformer 层数（默认：2）
- `n_heads`：注意力头数（默认：2）
- `hidden_size`：隐藏层维度（默认：64）
- `inner_size`：前馈网络维度（默认：256）
- `hidden_dropout_prob`：Dropout 概率（默认：0.5）
- `attn_dropout_prob`：注意力 Dropout 概率（默认：0.5）

### 多粒度参数

- `scales`：多尺度窗口大小（Tmall: [4, 20]，其他: [4, 10]）
- `user_level`：用户级别粒度（Tmall: [10, 2]，其他: [10, 4]）
- `item_level`：物品级别粒度（默认：3）
- `l_p`：池化参数（默认：4）

### 训练参数

- `MAX_ITEM_LIST_LENGTH`：最大序列长度（默认：200）
- `train_batch_size`：训练批大小（IJCAI: 24，其他: 64）
- `eval_batch_size`：评估批大小（IJCAI: 24，其他: 128）
- `mask_ratio`：掩码比例（默认：0.2）

### 聚合参数

- `agg`：聚合方式（默认：'con'）
- `agg_method`：聚合方法（默认：'maxpooling'）

## 评估指标

模型在以下指标上进行评估：

- **Recall@K**：召回率（K=5, 10, 101）
- **NDCG@K**：归一化折损累积增益（K=5, 10, 101）
- **MRR@K**：平均倒数排名（K=5, 10, 101）

主要评估指标：`NDCG@10`

## 项目结构

```
MGPT-2025/
├── run_M-GPT.py              # 主运行脚本
├── recbole/                  # RecBole 框架
│   ├── model/
│   │   └── sequential_recommender/
│   │       └── mgpt.py       # MGPT 模型实现
│   ├── data/                 # 数据处理模块
│   ├── trainer/              # 训练器模块
│   └── utils/                # 工具函数
├── dataset/                  # 数据集目录（需自行创建）
│   ├── tmall_beh/
│   ├── ijcai_beh/
│   └── retail_beh/
└── log/                      # 日志输出目录（自动创建）
```

## 复现论文结果

### Tmall 数据集

```bash
python run_M-GPT.py \
    --model MGPT \
    --dataset tmall_beh \
    --gpu_id 0
```

预期结果：
- Recall@10: ~0.08-0.10
- NDCG@10: ~0.06-0.08
- MRR@10: ~0.05-0.07

### IJCAI 数据集

```bash
python run_M-GPT.py \
    --model MGPT \
    --dataset ijcai_beh \
    --gpu_id 0
```

### Retail Rocket 数据集

```bash
python run_M-GPT.py \
    --model MGPT \
    --dataset retail_beh \
    --gpu_id 0
```

## 消融实验

模型支持消融实验，可以通过修改 `config_dict` 中的 `abaltion` 参数来进行：

```python
config_dict['abaltion'] = 'no_ms'  # 禁用多尺度
```

## 注意事项

1. **GPU 内存**：如果遇到 GPU 内存不足，可以：
   - 减小 `batch_size`
   - 减小 `hidden_size` 或 `inner_size`
   - 减小 `MAX_ITEM_LIST_LENGTH`

2. **训练时间**：完整训练一个数据集通常需要数小时到十几小时，取决于数据集大小和硬件配置。

3. **随机种子**：为了保证结果可复现，RecBole 会自动设置随机种子。

4. **日志输出**：训练日志会保存在 `log/` 目录下，可以查看详细的训练过程和最终结果。

## 引用

如果您使用了本代码，请引用我们的论文：

```bibtex
@inproceedings{MGPT2025,
  title={Multi-Grained Preference Enhanced Transformer for Multi-behavior Sequential Recommendation},
  author={Your Name},
  booktitle={Conference Name},
  year={2025}
}
```

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题，请通过 Issue 或邮件联系我们。
