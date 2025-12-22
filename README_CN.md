# PFHC: 面向AI工作负载模式预测的概率模糊超图卷积

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[English](README.md) | [中文](README_CN.md)

---

## 📋 项目概述

本仓库包含论文 **"面向AI工作负载模式预测的概率模糊超图卷积"** 的官方实现代码（论文投稿中）。

**PFHC** 是一个新颖的深度学习框架，专为AI基础设施即服务（AI-IaaS）环境中的工作负载模式预测而设计。该框架通过统一的概率-模糊不确定性建模方法，解决了非平稳工作负载演化、共居负载干扰和模糊模式转移等关键挑战。

### 🎯 核心特性

- **空间有向模糊超图卷积**：通过V→E和E→E有向卷积建模共居负载间的复杂资源竞争与干扰关系
- **条件因果模糊卷积**：采用波动感知的时序建模和动态感受野捕获非平稳模式转移规律
- **概率-模糊融合**：通过条件概率直觉模糊集统一概率转移分布与模糊隶属度
- **多数据集支持**：在真实云计算负载轨迹上验证，包括鹏城云脑、谷歌集群轨迹和阿里巴巴集群轨迹

---

## 🏗️ 框架架构

```
┌─────────────────────────────────────────────────────────────┐
│                    PFHC 框架                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │  空间有向模糊超图卷积                                  │  │
│  │  • V→E: 节点到超边聚合                                │  │
│  │  • E→E: 超边到超边干扰建模                            │  │
│  │  • 模糊化层                                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  条件因果模糊卷积                                      │  │
│  │  • 波动感知的感受野                                    │  │
│  │  • 动态条件卷积核生成                                  │  │
│  │  • 门控时序特征提取                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  概率-模糊关系融合预测                                 │  │
│  │  • 直觉模糊关系矩阵                                    │  │
│  │  • 温度缩放Softmax预测                                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 数据集

### 主要数据集（实验使用）

**鹏城云脑I集群数据集**
- **来源**：[启智OpenI平台](https://openi.pcl.ac.cn/potato/CloudBrain-datasets)
- **描述**：来自大规模GPU集群的真实AI训练工作负载轨迹
- **指标**：CPU利用率、GPU利用率、GPU显存、磁盘I/O
- **采样**：24小时窗口内15秒间隔采样

### 支持的数据集（框架兼容）

**谷歌集群轨迹（Google Cluster Trace）**
- **来源**：[Google Cluster Data](https://github.com/google/cluster-data)
- **描述**：来自谷歌计算集群的29天轨迹数据，包含12.5k台机器
- **说明**：框架通过数据适配器支持谷歌轨迹格式

**阿里巴巴集群轨迹（Alibaba Cluster Trace）**
- **来源**：[Alibaba Cluster Trace](https://github.com/alibaba/clusterdata)
- **描述**：来自阿里巴巴大规模集群的生产环境轨迹数据
- **说明**：框架通过数据适配器支持阿里巴巴轨迹格式

> **注意**：本研究主要在鹏城云脑数据集上进行验证。框架设计具有可扩展性，支持其他公开云工作负载数据集。

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+（用于GPU加速）
- 推荐16GB+内存

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/YOUR_USERNAME/PFHC.git
cd PFHC
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Windows系统: venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

### 以鹏城云脑数据集为例的数据集准备

#### 鹏城云脑数据集

1. **下载数据集**
```bash
# 从启智OpenI平台下载
# https://openi.pcl.ac.cn/potato/CloudBrain-datasets
```

2. **预处理原始数据**
```bash
python scripts/bake_dataset_v10.py
```

该脚本将：
- 提取工作负载特征（CPU、GPU、内存、I/O）
- 构建超图结构
- 生成训练/验证集划分
- 缓存预处理数据至 `data/baked_v10/`

### 模型训练

**使用默认配置训练：**
```bash
python scripts/train.py
```

**使用自定义参数训练：**
```bash
python scripts/train.py \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --num_fuzzy_rules 12 \
    --hnn_hidden 256 \
    --tcn_hidden 256
```

**关键超参数：**
- `--batch_size`：物理批次大小（默认：32）
- `--accumulation_steps`：梯度累积步数（默认：8）
- `--num_fuzzy_rules`：模糊模式数量（默认：12）
- `--temperature`：预测Softmax温度（默认：0.7）
- `--dropout`：Dropout比率（默认：0.20）

### 模型评估

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/pfhc_model_v10_final.pth \
    --test_data data/baked_v10/val/
```

---

## 📈 实验结果

### 鹏城云脑数据集上的性能对比

| 方法 | 准确率 | 精确率 | F1分数 | MAE |
|------|--------|--------|--------|-----|
| LSTM | 0.6234 | 0.6108 | 0.6045 | 0.1876 |
| GRU | 0.6512 | 0.6389 | 0.6321 | 0.1734 |
| TCN | 0.6789 | 0.6654 | 0.6598 | 0.1623 |
| GNN-LSTM | 0.7012 | 0.6891 | 0.6834 | 0.1512 |
| EvoGWP | 0.7234 | 0.7123 | 0.7089 | 0.1398 |
| **PFHC（本文）** | **0.7856** | **0.7734** | **0.7698** | **0.1124** |

*结果为5次不同随机种子运行的平均值*

### 消融实验

| 变体 | 准确率 | ΔAcc |
|------|--------|------|
| PFHC（完整） | 0.7856 | - |
| 无E→E卷积 | 0.7423 | -5.51% |
| 无条件因果 | 0.7312 | -6.92% |
| 无模糊融合 | 0.7189 | -8.49% |

---

## 📁 项目结构

```
PFHC/
├── pfhc/                          # 核心实现
│   ├── models/
│   │   └── model.py              # PFHC模型架构
│   ├── datasets/
│   │   ├── cloudbrain_dataset.py # 数据集加载器
│   │   ├── baked_dataset_v10.py  # 预处理数据集
│   │   └── collate.py            # 批次整理
│   ├── graphs/                    # 超图构建
│   └── utils/                     # 工具函数
├── scripts/
│   ├── train.py                   # 训练脚本
│   ├── evaluate.py                # 评估脚本
│   ├── bake_dataset_v10.py       # 数据预处理
│   └── calc_stats.py             # 统计计算
├── data/                          # 数据目录
│   ├── processed/                 # 原始处理数据
│   └── baked_v10/                # 缓存的预处理数据
├── checkpoints/                   # 模型检查点
├── logs/                          # 训练日志
├── requirements.txt               # Python依赖
├── LICENSE                        # MIT许可证
├── README.md                      # 英文README
└── README_CN.md                   # 本文件（中文）
```

---

## 🔧 依赖要求

```txt
torch>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
h5py>=3.7.0
tqdm>=4.62.0
pandas>=1.3.0
```

完整依赖列表见 `requirements.txt`。

---

## 📝 引用

如果本工作对您的研究有帮助，请引用：

```bibtex
@article{pfhc2025,
  title={Probabilistic Fuzzy Hypergraph Convolution for AI Workload Pattern Prediction},
  author={[论文发表后添加作者信息]},
  journal={[论文发表后添加期刊/会议信息]},
  year={2025},
  note={Paper under submission}
}
```

---

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- **鹏城实验室**提供云脑I数据集
- **谷歌**提供公开集群轨迹数据集
- **阿里巴巴**提供公开集群轨迹数据集
- 云计算和工作负载预测领域的所有贡献者和研究人员

---

## 📧 联系方式

如有问题和反馈，请在本仓库中提交issue。

---

## 🔗 相关资源

- [鹏城云脑数据集](https://openi.pcl.ac.cn/potato/CloudBrain-datasets)
- [谷歌集群轨迹](https://github.com/google/cluster-data)
- [阿里巴巴集群轨迹](https://github.com/alibaba/clusterdata)

---

**注意**：这是一个正在积极开发的研究项目。论文目前正在投稿中。代码和文档将持续更新。
