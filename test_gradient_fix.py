"""
测试梯度修复是否成功的脚本
"""
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader

from transformer_model import create_model
from tokenizer import ChineseTokenizer, SentimentDataset, load_data_from_csv
from train_transformer_web_fixed import run_training

def test_gradient():
    """测试梯度是否稳定"""
    print("=== 测试梯度稳定性 ===")

    # 加载数据
    train_texts, train_labels = load_data_from_csv('train.csv', text_column='text', label_column='label')

    # 创建tokenizer
    tokenizer = ChineseTokenizer(max_vocab_size=50000, use_word=True)
    tokenizer.build_vocab(train_texts, min_freq=1)

    # 创建简单数据集（不需要采样器）
    train_dataset = SentimentDataset(train_texts[:1000], train_labels[:1000], tokenizer, max_length=128)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    # 创建极简模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        num_classes=3,
        config={
            'd_model': 128,
            'num_heads': 2,
            'num_layers': 2,
            'd_ff': 256,
            'dropout': 0.1,
            'max_len': 128,
        }
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    print(f"设备: {device}")
    print(f"数据量: {len(train_dataset)}")
    print(f"标签分布: {Counter(train_labels[:1000])}")

    # 训练几个batch并监控梯度
    model.train()
    grad_history = []

    for i, batch in enumerate(train_loader):
        if i >= 10:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()

        # 计算梯度范数
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        grad_history.append(total_norm)

        print(f"Batch {i}: loss={loss.item():.4f}, grad_norm={total_norm:.4f}")

        # 检查是否有NaN或Inf
        if torch.isnan(logits).any():
            print("ERROR: logits包含NaN")
            break
        if torch.isinf(logits).any():
            print("ERROR: logits包含Inf")
            break

        optimizer.step()

    print(f"\n梯度历史: {[f'{g:.4f}' for g in grad_history]}")
    max_grad = max(grad_history)
    print(f"最大梯度: {max_grad:.4f}")

    if max_grad > 5.0:
        print("❌ 梯度仍然爆炸")
        return False
    elif max_grad > 1.0:
        print("⚠️ 梯度偏高但可接受")
        return True
    else:
        print("✅ 梯度稳定")
        return True

def test_training():
    """快速测试训练过程"""
    print("\n=== 测试训练过程 ===")

    config = {
        'train_data_path': 'train.csv',
        'val_data_path': 'validation.csv',
        'text_column': 'text',
        'label_column': 'label',
        'd_model': 128,
        'num_heads': 2,
        'num_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'max_length': 128,
        'batch_size': 32,
        'num_epochs': 2,  # 只训练2轮
        'learning_rate': 0.0001,
        'weight_decay': 0.01,
        'early_stopping_patience': 5,
        'max_vocab_size': 50000,
        'min_freq': 1,
        'use_word': True,
        'save_dir': 'test_checkpoint',
    }

    def progress_callback(progress):
        pass

    def log_callback(message):
        print(message)

    try:
        result = run_training(config, progress_callback, log_callback)
        print(f"训练完成！验证准确率: {result['val_acc']:.4f}")
        return True
    except Exception as e:
        print(f"训练失败: {e}")
        return False

if __name__ == "__main__":
    # 测试梯度
    grad_ok = test_gradient()

    # 测试训练
    train_ok = test_training()

    print("\n" + "="*50)
    if grad_ok and train_ok:
        print("✅ 所有测试通过！")
    else:
        print("❌ 还有问题需要解决")