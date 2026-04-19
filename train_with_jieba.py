"""
使用jieba分词改进的Transformer情感分析训练脚本
"""
import json
import os
import re
import time
from collections import Counter
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from transformer_model import create_model
from tokenizer import ChineseTokenizer, SentimentDataset, load_data_from_csv


# 优化后的训练配置
OPTIMIZED_CONFIG = {
    # 数据配置
    'data_path': 'data_augmented.csv',  # 使用数据增强集
    'text_column': 'sentence',
    'label_column': 'label',
    'test_size': 0.2,
    'random_state': 42,

    # 分词器配置 - 关键改进
    'use_word': True,           # 启用jieba词级分词
    'max_vocab_size': 50000,    # 扩大词汇表
    'min_freq': 1,               # 降低最小词频

    # 模型配置 - 适当增大
    'd_model': 384,             # 增大模型维度
    'num_heads': 8,
    'num_layers': 6,             # 增加层数
    'd_ff': 1024,                # 增大FFN维度
    'dropout': 0.15,
    'max_length': 128,

    # 训练配置
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 3e-4,       # 提高学习率
    'weight_decay': 0.01,
    'early_stopping_patience': 8,
    'label_smoothing': 0.05,

    # 保存目录
    'save_dir': 'transformer_checkpoints_jieba',
}


class ImprovedTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        save_dir='transformer_checkpoints_jieba',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
        }
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0

    def log(self, message: str):
        print(message)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return avg_loss, accuracy, f1

    def save_model(self, model_name):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, model_name))

    def train(self, num_epochs, early_stopping_patience=8):
        no_improve_count = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1 = self.evaluate()

            elapsed = time.time() - start_time

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            self.log(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f} | "
                f"time={elapsed:.2f}s"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.save_model('best_model.pt')
                self.log(f"[OK] 保存最佳模型，验证准确率={val_acc:.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                self.log(f"[INFO] 验证准确率未提升 ({no_improve_count}/{early_stopping_patience})")

            if no_improve_count >= early_stopping_patience:
                self.log("[INFO] 触发早停")
                break

        return {
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'history': self.history,
        }


def normalize_labels(labels):
    label_map_text = {
        'negative': 0,
        'positive': 1,
        'neutral': 2
    }

    normalized = []
    for x in labels:
        s = str(x).strip().lower()

        if s in label_map_text:
            normalized.append(label_map_text[s])
        elif s in {'0', '1', '2'}:
            normalized.append(int(s))
        else:
            raise ValueError(f"发现无法识别的标签值: {x}")

    return normalized


def run_training(config: Dict):
    os.makedirs(config['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    texts, labels = load_data_from_csv(
        config['data_path'],
        text_column=config['text_column'],
        label_column=config['label_column'],
    )

    # 清理文本
    texts = [re.sub(r'^\s*\d+[\.、,，]\s*', '', str(x)) for x in texts]

    # 标签标准化
    labels = normalize_labels(labels)

    # 创建分词器 - 使用jieba
    tokenizer = ChineseTokenizer(
        max_vocab_size=config['max_vocab_size'],
        use_word=config['use_word'],
    )
    tokenizer.build_vocab(texts, min_freq=config['min_freq'])
    tokenizer.save(os.path.join(config['save_dir'], 'tokenizer.pkl'))

    # 分割数据
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=config['test_size'], random_state=config['random_state'], stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.25, random_state=config['random_state'], stratify=train_labels
    )

    print(f"训练集: {len(train_texts)} 条")
    print(f"验证集: {len(val_texts)} 条")
    print(f"测试集: {len(test_texts)} 条")

    # 创建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=config['max_length'])
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length=config['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # 创建模型
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        num_classes=3,
        config={
            'd_model': config['d_model'],
            'num_heads': config['num_heads'],
            'num_layers': config['num_layers'],
            'd_ff': config['d_ff'],
            'dropout': config['dropout'],
            'max_len': config['max_length'],
        },
    ).to(device)

    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # 类别权重
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = []
    for i in range(3):
        count = label_counts.get(i, 1)
        class_weights.append(total_samples / (3 * count))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"类别权重: {class_weights.tolist()}")

    # 损失函数
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1, weight=None):
            super().__init__()
            self.smoothing = smoothing
            self.weight = weight

        def forward(self, input, target):
            log_probs = nn.functional.log_softmax(input, dim=-1)
            if self.weight is not None:
                log_probs = log_probs * self.weight[target].unsqueeze(1)
            nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
            smooth_loss = -log_probs.mean(dim=-1)
            loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
            return loss.mean()

    criterion = LabelSmoothingCrossEntropy(
        smoothing=config['label_smoothing'],
        weight=class_weights,
    )

    # 训练
    trainer = ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=config['save_dir'],
    )

    result = trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
    )

    # 测试集评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)

    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length=config['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"测试集准确率: {test_acc:.4f}")
    print(f"测试集F1分数: {test_f1:.4f}")
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['负面', '正面', '中性']))

    # 保存配置
    config_copy = config.copy()
    with open(os.path.join(config['save_dir'], 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_copy, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("使用jieba分词的Transformer情感分析训练")
    print("=" * 60)

    run_training(OPTIMIZED_CONFIG)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
