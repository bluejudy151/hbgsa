# 🚀 HBGSA 快速开始指南

5分钟快速上手 HBGSA 模型！

---

## ⚡ 快速开始（3步）

### 步骤 1: 环境准备

```bash
# 创建虚拟环境
conda create -n hbgsa python=3.9 -y
conda activate hbgsa

# 安装依赖
pip install torch numpy pandas scikit-learn scipy
```

### 步骤 2: 准备数据

确保数据文件夹结构正确（数据应该在 HBGSA 文件夹内）：

```
HBGSA/
├── retest.py              # 复现脚本
├── weights.pt             # 预训练权重文件
└── data/
    ├── data/              # 主数据集
    │   ├── affinity_data.csv
    │   ├── test_smi.csv
    │   └── test/
    │       ├── global/    # 蛋白质全局特征
    │       └── pocket/    # 蛋白质口袋特征
    └── CSAR_output/       # CSAR数据集
        ├── csar-affinity.csv
        ├── csar-smi.csv
        └── CSAR/
            ├── global/
            └── pocket/
```

### 步骤 3: 运行复现脚本

```bash
# 使用预训练权重评估模型
python retest.py
```

运行时会提示是否评估训练集和验证集：
- 输入 `y` - 评估所有数据集（训练集、验证集、测试集、CSAR）
- 输入 `n` - 只评估测试集和CSAR数据集（推荐，更快）

---

## 📊 预期结果

**测试集 (PDBbind Core Set)**:
- RMSE: ~1.19
- Pearson R: ~0.75+
- CI: ~0.80+

**CSAR 数据集**:
- RMSE: ~1.5-2.0
- Pearson R: ~0.60+

---

## 🔄 从头训练新模型

如果你想从头训练模型而不是使用预训练权重：

```bash
# 训练新模型
python main.py
```

训练需要完整的数据集，包括：
- `training_smi.csv`, `validation_smi.csv`, `test_smi.csv`
- `training/`, `validation/`, `test/` 文件夹
- `Bond/hbond_3d_flattened.csv` (氢键数据)

---

## 🔧 常见调整

### GPU 内存不足？

编辑 `retest.py` 开头的配置：
```python
BATCH_SIZE = 16  # 默认是 32，减小可以降低内存使用
```

### 修改数据路径？

编辑 `retest.py` 中的路径配置：
```python
DATA_PATH = SCRIPT_DIR.parent / 'data' / 'data'  # 主数据集路径
CSAR_DATA_PATH = SCRIPT_DIR.parent / 'data' / 'CSAR_output'  # CSAR数据集路径
```

---

## 📁 文件说明

### 核心文件
- `retest.py` - **复现脚本**，用于加载预训练权重并评估
- `weights.pt` - 预训练权重文件（交错格式，包含两个模型）
- `main.py` - 训练脚本
- `model.py` - 模型定义
- `dataset.py` - 数据加载器
- `trainer.py` - 训练器
- `config.py` - 配置文件

### 文档文件
- `README.md` - 项目说明
- `QUICKSTART.md` - 本文件，快速开始指南
- `复现指南.md` - 详细的复现步骤
- `SETUP.md` - 详细安装说明

---

## ❓ 遇到问题？

### 路径错误
确保数据文件夹在正确的位置：
- 主数据集：`HBGSA/data/data/`
- CSAR数据集：`HBGSA/data/CSAR_output/`
- 权重文件：`HBGSA/weights.pt`

### 权重加载错误
确保使用的是正确的权重文件 `weights.pt`，它包含两个模型的交错权重：
- 索引 0: 主数据集模型
- 索引 1: CSAR数据集模型

### 更多帮助
查看详细文档：
- [复现指南.md](复现指南.md) - 完整的复现步骤和常见问题
- [README.md](README.md) - 项目说明和模型架构
- [USE_HBOND重命名说明.md](USE_HBOND重命名说明.md) - 配置参数说明

---

## 📝 retest.py 使用示例

```bash
# 示例 1: 只评估测试集和CSAR（推荐）
$ python retest.py
Do you want to evaluate training and validation sets? (y/n)
Enter: n

# 示例 2: 评估所有数据集
$ python retest.py
Do you want to evaluate training and validation sets? (y/n)
Enter: y
```

---

**就这么简单！开始你的药物发现之旅吧！** 🎯
