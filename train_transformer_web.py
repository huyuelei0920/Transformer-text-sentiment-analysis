"""
纯 Transformer 情感分析模型训练脚本
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from transformer_model import create_model
from tokenizer import ChineseTokenizer, SentimentDataset, load_data_from_csv


# 训练器
class StreamlitTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        save_dir='transformer_checkpoints',
        progress_callback: Optional[Callable[[float], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.progress_callback = progress_callback
        self.log_callback = log_callback
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

    # 日志输出
    def log(self, message: str):
        print(message)
        if self.log_callback:
            self.log_callback(message)

    # 单轮训练
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

        avg_loss = total_loss / max(len(self.train_loader), 1)
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        return avg_loss, accuracy

    # 验证
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

        avg_loss = total_loss / max(len(self.val_loader), 1)
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else 0.0
        return avg_loss, accuracy, macro_f1

    # 保存模型
    def save_model(self, filename: str):
        path = os.path.join(self.save_dir, filename)
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_acc': self.best_val_acc,
                'best_val_f1': self.best_val_f1,
                'history': self.history,
            },
            path,
        )

    # 训练主过程
    def train(self, num_epochs: int, early_stopping_patience: int = 8):
        no_improve_count = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1 = self.evaluate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            elapsed = time.time() - start_time

            self.log(
                f"Epoch {epoch + 1}/{num_epochs}\n"
                f"train_loss = {train_loss:.4f}\n"
                f"train_acc  = {train_acc:.4f}\n"
                f"val_loss   = {val_loss:.4f}\n"
                f"val_acc    = {val_acc:.4f}\n"
                f"val_f1     = {val_f1:.4f}\n"
                f"time       = {elapsed:.2f}s"
            )

            if self.progress_callback:
                self.progress_callback((epoch + 1) / num_epochs)

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


# 标签标准化
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


# 数据集校验
def validate_dataset(df, text_column: str, label_column: str):
    if text_column not in df.columns:
        raise ValueError(f"文本列不存在: {text_column}")
    if label_column not in df.columns:
        raise ValueError(f"标签列不存在: {label_column}")
    if df[text_column].isnull().any():
        raise ValueError("文本列存在空值，请先清理")
    if df[label_column].isnull().any():
        raise ValueError("标签列存在空值，请先清理")

    label_values = set(df[label_column].astype(str).str.strip().str.lower().tolist())
    valid_numeric = {"0", "1", "2"}
    valid_text = {"negative", "positive", "neutral"}

    if not (label_values.issubset(valid_numeric) or label_values.issubset(valid_text)):
        raise ValueError(
            f"标签列只能包含 0/1/2 或 negative/positive/neutral，当前检测到: {sorted(label_values)}"
        )


# 训练入口
def run_training(
    config: Dict,
    progress_callback: Optional[Callable[[float], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
):
    os.makedirs(config['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if log_callback:
        log_callback(f"使用设备: {device}")

    texts, labels = load_data_from_csv(
        config['data_path'],
        text_column=config['text_column'],
        label_column=config['label_column'],
    )

    # 清理文本开头的编号前缀
    texts = [re.sub(r'^\s*\d+[\.、,，]\s*', '', str(x)) for x in texts]

    # 标签统一映射为数字
    labels = normalize_labels(labels)

    tokenizer = ChineseTokenizer(
        max_vocab_size=config['max_vocab_size'],
        use_word=config['use_word'],
    )
    tokenizer.build_vocab(texts, min_freq=config['min_freq'])
    tokenizer.save(os.path.join(config['save_dir'], 'tokenizer.pkl'))

    # 分割数据为训练集、验证集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=config['test_size'], random_state=config['random_state'], stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.25, random_state=config['random_state'], stratify=train_labels
    )

    if log_callback:
        log_callback(f"训练集: {len(train_texts)} 条")
        log_callback(f"验证集: {len(val_texts)} 条")
        log_callback(f"测试集: {len(test_texts)} 条")

    train_dataset = SentimentDataset(
        train_texts, train_labels, tokenizer, max_length=config['max_length']
    )
    val_dataset = SentimentDataset(
        val_texts, val_labels, tokenizer, max_length=config['max_length']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    model = create_model(
        vocab_size=tokenizer.vocab_size,
        num_classes=3,
        config={
            'd_model': config.get('d_model', 256),
            'num_heads': config.get('num_heads', 8),
            'num_layers': config.get('num_layers', 4),
            'd_ff': config.get('d_ff', 512),
            'dropout': config.get('dropout', 0.15),
            'max_len': config['max_length'],
        },
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01),
    )

    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    # 计算类别权重
    class_weights = []
    for i in range(3):
        count = label_counts.get(i, 1)
        class_weights.append(total_samples / (3 * count))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 使用带标签平滑的交叉熵损失
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
        smoothing=config.get('label_smoothing', 0.05),
        weight=class_weights,
    )

    trainer = StreamlitTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=config['save_dir'],
        progress_callback=progress_callback,
        log_callback=log_callback,
    )

    result = trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
    )

    # 使用测试集评估模型性能
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    plt.rcParams['axes.unicode_minus'] = False

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
        f1_score,
    )

    test_dataset = SentimentDataset(
        test_texts, test_labels, tokenizer, max_length=config['max_length']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')

    if log_callback:
        log_callback("\n" + "=" * 60)
        log_callback("测试集评估结果")
        log_callback("=" * 60)
        log_callback(f"测试集准确率: {test_acc:.4f}")
        log_callback(f"测试集F1分数: {test_f1:.4f}")
        log_callback("\n分类报告:")
        log_callback(classification_report(all_labels, all_preds, target_names=['负面', '正面', '中性']))

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    cm_labels = ['负面', '正面', '中性']
    if log_callback:
        log_callback("\n混淆矩阵 (测试集):")
        log_callback(str(cm))

    cm_path = os.path.join(config['save_dir'], 'confusion_matrix.png')
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels).plot(
        ax=ax, cmap='Blues', colorbar=True
    )
    ax.set_title('混淆矩阵（测试集）')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 保存配置
    with open(os.path.join(config['save_dir'], 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'd_model': config['d_model'],
                'num_heads': config['num_heads'],
                'num_layers': config['num_layers'],
                'd_ff': config['d_ff'],
                'dropout': config['dropout'],
                'max_length': config['max_length'],
                'test_acc': test_acc,
                'test_f1': test_f1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 更新结果
    result['test_acc'] = test_acc
    result['test_f1'] = test_f1
    result['confusion_matrix_path'] = cm_path

    return result