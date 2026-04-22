# 基于Transformer的文本情感分析系统设计与实现

Design and implementation of a Transformer-based text sentiment analysis system for my thesis project.

## 项目简介

本项目是一个**基于纯 Transformer 架构**的中文情感分析系统，从零实现了 Transformer Encoder 用于文本分类任务。系统支持三分类情感识别：正面、负面、中性。

## 特性

- ✅ **纯 Transformer 架构** - 从零实现，不依赖预训练模型
- ✅ **多头自注意力机制** - 8个注意力头并行计算
- ✅ **位置编码** - 正弦/余弦固定位置编码
- ✅ **Web可视化界面** - 基于 Streamlit 的交互式界面
- ✅ **批量分析** - 支持批量文本情感分析
- ✅ **可视化图表** - Plotly 交互式图表展示结果

## 模型架构

| 参数 | 值 |
|------|-----|
| d_model | 256 |
| num_heads | 8 |
| num_layers | 4 |
| d_ff | 512 |
| dropout | 0.1 |
| 总参数量 | ~470万 |

## 快速开始

### 安装依赖

```bash
pip install torch pandas numpy scikit-learn streamlit plotly tqdm
```

### 训练模型

```bash
python train_transformer_web.py
```

### 启动 Web 应用

```bash
streamlit run app_transformer_web.py
# 或双击 启动界面.bat
```

### 命令行推理

```bash
# 演示模式
python inference.py

# 分析单个文本
python inference.py --text "今天天气真好"

# 交互模式
python inference.py --interactive
```

## 项目结构

```
├── transformer_model.py      # Transformer 模型架构
├── tokenizer.py              # 中文分词器和数据处理
├── train_transformer_web.py  # 训练脚本
├── inference.py              # 模型推理接口
├── app_transformer_web.py    # Streamlit Web 应用
├── 使用说明.md                # 详细使用说明
├── train.csv                 # 训练数据
├── validation.csv            # 验证数据
├── transformer_checkpoints/  # 模型保存目录
│   ├── best_model.pt        # 最佳模型权重
│   ├── tokenizer.pkl        # 分词器
│   └── config.json          # 配置文件
└── transformer_checkpoints_jieba/  # 结巴分词版本模型
```

## 技术栈

- **深度学习框架**: PyTorch
- **Web框架**: Streamlit
- **可视化**: Plotly
- **数据处理**: Pandas, NumPy
- **机器学习工具**: Scikit-learn

## 模型架构详解

### 1. 位置编码 (Positional Encoding)

使用正弦和余弦函数生成固定位置编码：

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 2. 多头自注意力 (Multi-Head Attention)

- 将输入通过 Q、K、V 线性变换
- 分成多个头并行计算注意力
- 缩放点积注意力：Attention(Q,K,V) = softmax(QK^T/√d_k)V

### 3. 前馈网络 (Feed-Forward Network)

```
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
```

### 4. 编码器层 (Transformer Encoder Layer)

```
x = LayerNorm(x + MultiHeadAttention(x))
x = LayerNorm(x + FeedForward(x))
```

## 使用示例

### Python API

```python
from inference import create_predictor

# 创建预测器
predictor = create_predictor(
    model_dir='transformer_checkpoints',
    model_file='best_model.pt'
)

# 预测单个文本
result = predictor.predict("今天天气真好，心情很愉快")
print(f"情感: {result['label_name']}")
print(f"置信度: {result['confidence_percent']}")

# 批量预测
texts = ["这个产品很好", "太差了", "一般般"]
results = predictor.predict_batch(texts)
```

## 许可证

MIT License

## 致谢

- Transformer架构参考: "Attention Is All You Need" (Vaswani et al., 2017)
