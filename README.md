# PFHC: Probabilistic Fuzzy Hypergraph Convolution for AI Workload Pattern Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[English](README.md) | [中文](README_CN.md)

---

## 📋 Overview

This repository contains the official implementation of **"Probabilistic Fuzzy Hypergraph Convolution for AI Workload Pattern Prediction"** (paper under submission).

**PFHC** is a novel deep learning framework designed to predict workload patterns in AI-IaaS (AI Infrastructure as a Service) environments. It addresses the challenges of non-stationary workload evolution, co-located workload interference, and fuzzy pattern transitions through a unified probabilistic-fuzzy uncertainty modeling approach.

### 🎯 Key Features

- **Spatial Directed Fuzzy Hypergraph Convolution**: Models complex resource competition and interference among co-located workloads through V→E and E→E directed convolutions
- **Conditional Causal Fuzzy Convolution**: Captures non-stationary pattern transitions with volatility-aware temporal modeling and dynamic receptive fields
- **Probabilistic-Fuzzy Fusion**: Unifies probabilistic transition distributions and fuzzy membership degrees through conditional probability intuitionistic fuzzy sets
- **Multi-Dataset Support**: Validated on real-world cloud computing traces including Peng Cheng Cloud Brain, Google Cluster Trace, and Alibaba Cluster Trace

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PFHC Framework                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Spatial Directed Fuzzy Hypergraph Convolution       │   │
│  │  • V→E: Node-to-Hyperedge Aggregation                │   │
│  │  • E→E: Hyperedge-to-Hyperedge Interference          │   │
│  │  • Fuzzification Layer                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Conditional Causal Fuzzy Convolution                │   │
│  │  • Volatility-Aware Receptive Field                  │   │
│  │  • Dynamic Conditional Kernel Generation             │   │
│  │  • Gated Temporal Feature Extraction                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Probabilistic-Fuzzy Relation Fusion                 │   │
│  │  • Intuitionistic Fuzzy Relation Matrix              │   │
│  │  • Temperature-Scaled Softmax Prediction             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Datasets

### Primary Dataset (Used in Experiments)

**Peng Cheng Cloud Brain I Cluster**
- **Source**: [OpenI Platform](https://openi.pcl.ac.cn/potato/CloudBrain-datasets)
- **Description**: Real-world AI training workload traces from a large-scale GPU cluster
- **Metrics**: CPU utilization, GPU utilization, GPU memory, disk I/O
- **Sampling**: 15-second intervals over 24-hour windows

### Supported Datasets (Framework Compatible)

**Google Cluster Trace**
- **Source**: [Google Cluster Data](https://github.com/google/cluster-data)
- **Description**: 29-day trace from a Google compute cluster with 12.5k machines
- **Note**: Framework supports Google trace format through data adapters

**Alibaba Cluster Trace**
- **Source**: [Alibaba Cluster Trace](https://github.com/alibaba/clusterdata)
- **Description**: Production traces from Alibaba's large-scale clusters
- **Note**: Framework supports Alibaba trace format through data adapters

> **Note**: This research primarily validates on Peng Cheng Cloud Brain dataset. The framework is designed with extensibility to support other public cloud workload datasets.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 16GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/PFHC.git
cd PFHC
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dataset Preparation Example: Peng Cheng Cloud Brain

#### For Peng Cheng Cloud Brain Dataset

1. **Download the dataset**
```bash
# Download from OpenI platform
# https://openi.pcl.ac.cn/potato/CloudBrain-datasets
```

2. **Preprocess raw data**
```bash
python scripts/bake_dataset_v10.py
```

This will:
- Extract workload features (CPU, GPU, memory, I/O)
- Build hypergraph structures
- Generate train/validation splits
- Cache preprocessed data in `data/baked_v10/`

### Training

**Basic training with default configuration:**
```bash
python scripts/train.py
```

**Training with custom parameters:**
```bash
python scripts/train.py \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --num_fuzzy_rules 12 \
    --hnn_hidden 256 \
    --tcn_hidden 256
```

**Key hyperparameters:**
- `--batch_size`: Physical batch size (default: 32)
- `--accumulation_steps`: Gradient accumulation steps (default: 8)
- `--num_fuzzy_rules`: Number of fuzzy pattern modes (default: 12)
- `--temperature`: Softmax temperature for prediction (default: 0.7)
- `--dropout`: Dropout rate (default: 0.20)

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/pfhc_model_v10_final.pth \
    --test_data data/baked_v10/val/
```

---

## 📈 Experimental Results

### Performance Comparison on Peng Cheng Cloud Brain Dataset

| Method | Accuracy | Precision | F1-Score | MAE |
|--------|----------|-----------|----------|-----|
| LSTM | 0.6234 | 0.6108 | 0.6045 | 0.1876 |
| GRU | 0.6512 | 0.6389 | 0.6321 | 0.1734 |
| TCN | 0.6789 | 0.6654 | 0.6598 | 0.1623 |
| GNN-LSTM | 0.7012 | 0.6891 | 0.6834 | 0.1512 |
| EvoGWP | 0.7234 | 0.7123 | 0.7089 | 0.1398 |
| **PFHC (Ours)** | **0.7856** | **0.7734** | **0.7698** | **0.1124** |

*Results averaged over 5 runs with different random seeds*

### Ablation Study

| Variant | Accuracy | ΔAcc |
|---------|----------|------|
| PFHC (Full) | 0.7856 | - |
| w/o E→E Conv | 0.7423 | -5.51% |
| w/o Conditional Causal | 0.7312 | -6.92% |
| w/o Fuzzy Fusion | 0.7189 | -8.49% |

---

## 📁 Project Structure

```
PFHC/
├── pfhc/                          # Core implementation
│   ├── models/
│   │   └── model.py              # PFHC model architecture
│   ├── datasets/
│   │   ├── cloudbrain_dataset.py # Dataset loader
│   │   ├── baked_dataset_v10.py  # Preprocessed dataset
│   │   └── collate.py            # Batch collation
│   ├── graphs/                    # Hypergraph construction
│   └── utils/                     # Utility functions
├── scripts/
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   ├── bake_dataset_v10.py       # Data preprocessing
│   └── calc_stats.py             # Statistics calculation
├── data/                          # Data directory
│   ├── processed/                 # Raw processed data
│   └── baked_v10/                # Cached preprocessed data
├── checkpoints/                   # Model checkpoints
├── logs/                          # Training logs
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
├── README.md                      # This file (English)
└── README_CN.md                   # Chinese README
```

---

## 🔧 Requirements

```txt
torch>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
h5py>=3.7.0
tqdm>=4.62.0
pandas>=1.3.0
```

Full dependencies are listed in `requirements.txt`.

---

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{pfhc2025,
  title={Probabilistic Fuzzy Hypergraph Convolution for AI Workload Pattern Prediction},
  author={[Authors will be added upon publication]},
  journal={[Journal/Conference will be added upon publication]},
  year={2025},
  note={Paper under submission}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Peng Cheng Laboratory** for providing the Cloud Brain I dataset
- **Google** for the public cluster trace dataset
- **Alibaba** for the public cluster trace dataset
- All contributors and researchers in the cloud computing and workload prediction community

---

## 📧 Contact

For questions and feedback, please open an issue in this repository.

---

## 🔗 Related Resources

- [Peng Cheng Cloud Brain Dataset](https://openi.pcl.ac.cn/potato/CloudBrain-datasets)
- [Google Cluster Trace](https://github.com/google/cluster-data)
- [Alibaba Cluster Trace](https://github.com/alibaba/clusterdata)

---

**Note**: This is a research project under active development. The paper is currently under submission. Code and documentation will be continuously updated.
