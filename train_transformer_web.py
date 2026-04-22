"""
纯 Transformer 情感分析模型训练脚本
"""
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

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
        vocab_size,
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
        self.vocab_size = vocab_size
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
        batch_count = 0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)

            # 检查数值稳定性
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                continue

            loss = self.criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            batch_count += 1

        if batch_count == 0:
            return 0.0, 0.0

        avg_loss = total_loss / batch_count
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    # 验证
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        batch_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue

                loss = self.criterion(logits, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                batch_count += 1

        if batch_count == 0:
            return 0.0, 0.0, 0.0

        avg_loss = total_loss / batch_count
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        return avg_loss, accuracy, macro_f1

    # 保存模型
    def save_model(self, filename: str):
        path = os.path.join(self.save_dir, filename)

        if os.path.exists(path):
            try:
                os.remove(path)
                time.sleep(0.1)
            except Exception:
                pass

        temp_path = path + '.tmp'
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_acc': self.best_val_acc,
                'best_val_f1': self.best_val_f1,
                'vocab_size': self.vocab_size,
                'history': self.history,
            },
            temp_path,
        )
        import shutil
        shutil.move(temp_path, path)

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
            current_lr = self.optimizer.param_groups[0]['lr']

            self.log(
                f"Epoch {epoch + 1}/{num_epochs}\n"
                f"train_loss = {train_loss:.4f}\n"
                f"train_acc  = {train_acc:.4f}\n"
                f"val_loss   = {val_loss:.4f}\n"
                f"val_acc    = {val_acc:.4f}\n"
                f"val_f1     = {val_f1:.4f}\n"
                f"lr         = {current_lr:.6f}\n"
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


# 下采样平衡数据
def balance_by_downsampling(texts, labels, seed=42):
    """对多数类下采样，使各类样本数等于非最大类中的最大值"""
    rng = random.Random(seed)

    groups = defaultdict(list)
    for text, label in zip(texts, labels):
        groups[label].append((text, label))

    counts = {k: len(v) for k, v in groups.items()}
    max_count = max(counts.values())
    # 目标数量 = 非最大类中的最大值
    non_max = [c for c in counts.values() if c < max_count]
    target = max(non_max) if non_max else max_count

    balanced = []
    for label, items in groups.items():
        if len(items) > target:
            sampled = rng.sample(items, target)
        else:
            sampled = list(items)
        balanced.extend(sampled)

    rng.shuffle(balanced)
    texts_out = [t for t, _ in balanced]
    labels_out = [l for _, l in balanced]

    new_counts = Counter(labels_out)
    print(f"下采样: {dict(counts)} -> {dict(new_counts)}，共 {len(texts_out)} 条")
    return texts_out, labels_out


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

    # 分别加载训练集和验证集
    train_texts, train_labels = load_data_from_csv(
        config['train_data_path'],
        text_column=config['text_column'],
        label_column=config['label_column'],
    )

    val_texts, val_labels = load_data_from_csv(
        config['val_data_path'],
        text_column=config['text_column'],
        label_column=config['label_column'],
    )

    # 清理文本开头的编号前缀
    train_texts = [re.sub(r'^\s*\d+[\.、,，]\s*', '', str(x)) for x in train_texts]
    val_texts = [re.sub(r'^\s*\d+[\.、,，]\s*', '', str(x)) for x in val_texts]

    # 标签统一映射为数字
    train_labels = normalize_labels(train_labels)
    val_labels = normalize_labels(val_labels)

    # 下采样平衡数据
    if config.get('balance_method', 'downsample') == 'downsample':
        if log_callback:
            log_callback(f"下采样前训练集: {len(train_texts)} 条，{dict(Counter(train_labels))}")
            log_callback(f"下采样前验证集: {len(val_texts)} 条，{dict(Counter(val_labels))}")
        train_texts, train_labels = balance_by_downsampling(train_texts, train_labels)
        val_texts, val_labels = balance_by_downsampling(val_texts, val_labels)
        if log_callback:
            log_callback(f"下采样后训练集: {len(train_texts)} 条")
            log_callback(f"下采样后验证集: {len(val_texts)} 条")

    tokenizer = ChineseTokenizer(
        max_vocab_size=config['max_vocab_size'],
        use_word=config['use_word'],
    )
    tokenizer.build_vocab(train_texts, min_freq=config['min_freq'])
    tokenizer.save(os.path.join(config['save_dir'], 'tokenizer.pkl'))

    if log_callback:
        log_callback(f"训练集: {len(train_texts)} 条")
        log_callback(f"验证集: {len(val_texts)} 条")

    # 不使用采样器，直接使用标准训练
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

    # 模型配置（优先从 config 读取，否则用默认值）
    model_config = {
        'd_model': config.get('d_model', 256),
        'num_heads': config.get('num_heads', 8),
        'num_layers': config.get('num_layers', 4),
        'd_ff': config.get('d_ff', 512),
        'dropout': config.get('dropout', 0.2),
        'max_len': config.get('max_length', 128),
    }

    model = create_model(
        vocab_size=tokenizer.vocab_size,
        num_classes=3,
        config=model_config,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
    )

    # 添加学习率调度器（更简单）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )

    # 标签平滑交叉熵损失，防止过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainer = StreamlitTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        vocab_size=tokenizer.vocab_size,
        save_dir=config['save_dir'],
        progress_callback=progress_callback,
        log_callback=log_callback,
    )

    # 自定义训练循环，支持scheduler
    num_epochs = config['num_epochs']
    early_stopping_patience = config['early_stopping_patience']
    no_improve_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc, val_f1 = trainer.evaluate()

        # 更新学习率
        scheduler.step(val_acc)
        current_lr = trainer.optimizer.param_groups[0]['lr']

        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['train_acc'].append(train_acc)
        trainer.history['val_acc'].append(val_acc)
        trainer.history['val_f1'].append(val_f1)

        elapsed = time.time() - start_time

        log_msg = (
            f"Epoch {epoch + 1}/{num_epochs}\n"
            f"train_loss = {train_loss:.4f}\n"
            f"train_acc  = {train_acc:.4f}\n"
            f"val_loss   = {val_loss:.4f}\n"
            f"val_acc    = {val_acc:.4f}\n"
            f"val_f1     = {val_f1:.4f}\n"
            f"lr         = {current_lr:.6f}\n"
            f"time       = {elapsed:.2f}s"
        )
        trainer.log(log_msg)

        if trainer.progress_callback:
            trainer.progress_callback((epoch + 1) / num_epochs)

        if val_acc > trainer.best_val_acc:
            trainer.best_val_acc = val_acc
            trainer.best_val_f1 = val_f1
            trainer.save_model('best_model.pt')
            trainer.log(f"[OK] 保存最佳模型，验证准确率={val_acc:.4f}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            trainer.log(f"[INFO] 验证准确率未提升 ({no_improve_count}/{early_stopping_patience})")

        if no_improve_count >= early_stopping_patience:
            trainer.log("[INFO] 触发早停")
            break

    # 保存配置
    result = {
        'best_val_acc': trainer.best_val_acc,
        'best_val_f1': trainer.best_val_f1,
        'val_acc': trainer.best_val_acc,
        'val_f1': trainer.best_val_f1,
        'history': trainer.history,
    }

    # 使用验证集评估模型性能
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

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    if log_callback:
        log_callback("\n" + "=" * 60)
        log_callback("验证集最终评估结果")
        log_callback("=" * 60)
        log_callback(f"验证集准确率: {val_acc:.4f}")
        log_callback(f"验证集F1分数: {val_f1:.4f}")
        log_callback("\n分类报告:")
        log_callback(classification_report(all_labels, all_preds, target_names=['负面', '正面', '中性']))

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    cm_labels = ['负面', '正面', '中性']
    if log_callback:
        log_callback("\n混淆矩阵 (验证集):")
        log_callback(str(cm))

    cm_path = os.path.join(config['save_dir'], 'confusion_matrix.png')
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels).plot(
        ax=ax, cmap='Blues', colorbar=True
    )
    ax.set_title('混淆矩阵（验证集）')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 保存配置
    with open(os.path.join(config['save_dir'], 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'd_model': model_config['d_model'],
                'num_heads': model_config['num_heads'],
                'num_layers': model_config['num_layers'],
                'd_ff': model_config['d_ff'],
                'dropout': model_config['dropout'],
                'max_length': config['max_length'],
                'val_acc': val_acc,
                'val_f1': val_f1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    result['val_acc'] = val_acc
    result['val_f1'] = val_f1
    result['confusion_matrix_path'] = cm_path

    return result
