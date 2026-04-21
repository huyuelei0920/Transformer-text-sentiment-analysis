"""
诊断梯度爆炸的脚本
"""
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

from transformer_model import create_model
from tokenizer import ChineseTokenizer, SentimentDataset, load_data_from_csv

# 加载数据
print("=== 加载数据 ===")
train_texts, train_labels = load_data_from_csv('train.csv', text_column='text', label_column='label')

print(f"\n训练集: {len(train_texts)} 条")
print(f"标签分布: {Counter(train_labels)}")

# 创建tokenizer
tokenizer = ChineseTokenizer(max_vocab_size=50000, use_word=True)
tokenizer.build_vocab(train_texts, min_freq=1)

print(f"词表大小: {tokenizer.vocab_size}")

# 创建数据集和采样器
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=128)

label_counts = Counter(train_labels)
sample_weights = [1.0 / label_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

model = create_model(
    vocab_size=tokenizer.vocab_size,
    num_classes=3,
    config={
        'd_model': 128,
        'num_heads': 2,
        'num_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_len': 128
    }
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)

print(f"\n=== 开始梯度诊断 ===")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练几个batch并记录梯度
model.train()
for step, batch in enumerate(train_loader):
    if step >= 10:  # 只检查前10个batch
        break

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)

    # 前向传播
    logits = model(input_ids, attention_mask)
    loss = criterion(logits, labels)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 检查梯度
    grad_norms = {}
    max_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
            max_grad = max(max_grad, grad_norm)

    # 打印信息
    print(f"\nStep {step}:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits范围: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"  最大梯度: {max_grad:.3f}")
    print(f"  梯度范数(各层):")

    for name in ['embedding.weight', 'encoder.layers.0.self_attention.W_q.weight',
                 'encoder.layers.0.feed_forward.linear1.weight', 'classifier.0.weight']:
        if name in grad_norms:
            print(f"    {name}: {grad_norms[name]:.3f}")

    # 检查是否有异常梯度
    if max_grad > 1.0:
        print(f"  ⚠️ 梯度过大！可能原因:")
        print(f"    1. 某些参数初始化不当")
        print(f"    2. 数据中有异常样本")
        print(f"    3. 模型架构问题")

    optimizer.step()

print("\n=== 诊断完成 ===")
print("\n建议:")
if max_grad > 1.0:
    print("- 梯度仍然爆炸，尝试:")
    print("  1. 进一步降低学习率 (1e-5)")
    print("  2. 检查数据质量")
    print("  3. 使用梯度累积而不是大batch")
else:
    print("- 梯度正常，可以开始正常训练")
