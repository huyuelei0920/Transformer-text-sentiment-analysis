import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 使用系统内支持中文的黑体/雅黑，否则标题与类别名会显示为方框
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
import json
import time

from transformer_model import create_model
from tokenizer import ChineseTokenizer, SentimentDataset, load_data_from_csv


class Trainer:
    """
    训练器类
    """

    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            save_dir='transformer_checkpoints',
            num_classes=3
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            save_dir: 保存目录
            num_classes: 类别数量
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.num_classes = num_classes

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 记录训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': []
        }

        # 最佳模型跟踪
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0

    def train_epoch(self):
        """
        训练一个 epoch
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        # 使用 tqdm 进度条，配置适合 Windows 的参数
        progress_bar = tqdm(
            self.train_loader,
            desc="训练中",
            ncols=100,  # 固定宽度
            leave=True,  # 保留进度条
            file=None  # 使用标准输出
        )

        for batch in progress_bar:
            # 获取数据
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            logits = self.model(input_ids, attention_mask)

            # 计算损失
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条显示
            avg_loss = total_loss / (len(all_labels) // self.train_loader.batch_size + 1)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{avg_loss:.4f}'})

        # 计算指标
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def evaluate(self):
        """
        评估模型
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        # 使用 tqdm 进度条
        progress_bar = tqdm(
            self.val_loader,
            desc="验证中",
            ncols=100,
            leave=True
        )

        with torch.no_grad():
            for batch in progress_bar:
                # 获取数据
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                logits = self.model(input_ids, attention_mask)

                # 计算损失
                loss = self.criterion(logits, labels)

                # 记录
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, accuracy, f1, all_labels, all_preds

    def train(self, num_epochs, early_stopping_patience=5):
        """
        训练模型

        Args:
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
        """
        print(f"\n开始训练，共 {num_epochs} 轮")
        print(f"设备: {self.device}")
        print(f"早停耐心值: {early_stopping_patience} (验证准确率连续{early_stopping_patience}轮未提升则停止)")
        print("=" * 60)

        no_improve_count = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc, val_f1, _, _ = self.evaluate()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            epoch_time = time.time() - start_time

            # 打印结果
            print(f"\n训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f} | F1: {val_f1:.4f}")
            print(f"耗时: {epoch_time:.2f} 秒")

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.save_model('best_model.pt')
                print(f"[OK] 保存最佳模型 (准确率: {val_acc:.4f})")
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"[INFO] 验证准确率未提升 ({no_improve_count}/{early_stopping_patience})")

            # 早停
            if no_improve_count >= early_stopping_patience:
                print(f"\n早停触发：验证准确率连续 {early_stopping_patience} 轮未提升")
                print(f"最佳验证准确率: {self.best_val_acc:.4f}")
                break

        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"最佳验证 F1: {self.best_val_f1:.4f}")

    def save_model(self, filename):
        """
        保存模型
        """
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }, path)

    def load_model(self, filename):
        """
        加载模型
        """
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.history = checkpoint['history']
        print(f"模型已从 {path} 加载")

    def plot_history(self, save_path=None):
        """
        绘制训练历史
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 损失曲线
        axes[0].plot(self.history['train_loss'], label='训练损失')
        axes[0].plot(self.history['val_loss'], label='验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('损失曲线')
        axes[0].legend()
        axes[0].grid(True)

        # 准确率曲线
        axes[1].plot(self.history['train_acc'], label='训练准确率')
        axes[1].plot(self.history['val_acc'], label='验证准确率')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('准确率曲线')
        axes[1].legend()
        axes[1].grid(True)

        # F1 曲线
        axes[2].plot(self.history['val_f1'], label='验证 F1', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('F1 分数曲线')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存到 {save_path}")

        plt.close()

    def print_classification_report(self):
        """
        打印分类报告
        """
        _, _, _, labels, preds = self.evaluate()

        print("\n分类报告:")
        print("=" * 60)

        target_names = ['负面 (Negative)', '正面 (Positive)', '中性 (Neutral)']
        print(classification_report(labels, preds, target_names=target_names))

        # 混淆矩阵（验证集；固定标签顺序，保证 3×3）
        cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
        print("\n混淆矩阵:")
        print(cm)

        cm_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(
            ax=ax, cmap='Blues', colorbar=True
        )
        ax.set_title('混淆矩阵（验证集）')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n混淆矩阵图已保存到 {cm_path}")


def main():
    """
    主训练函数
    """
    # 配置
    CONFIG = {
        # ===== 默认超参：在泛化与拟合能力之间折中（可按验证集再微调）=====

        'data_path': 'data_shopping.csv',
        'text_column': 'sentence',
        'label_column': 'label',

        # 模型（d_model 需能被 num_heads 整除）
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,
        'd_ff': 512,
        'dropout': 0.15,
        'max_length': 128,

        # 优化：略减轻正则 + 给足训练轮数与早停耐心，减少欠拟合
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'early_stopping_patience': 10,
        'label_smoothing': 0.05,

        # 词表：略放宽以保留更多低频字/词
        'max_vocab_size': 22000,
        'min_freq': 2,
        'use_word': False,

        'save_dir': 'transformer_checkpoints'
    }

    print("=" * 60)
    print("纯 Transformer 情感分析模型训练")
    print("=" * 60)

    # 打印配置
    print("\n配置:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n" + "=" * 60)
    print("加载数据")
    print("=" * 60)

    texts, labels = load_data_from_csv(
        CONFIG['data_path'],
        text_column=CONFIG['text_column'],
        label_column=CONFIG['label_column']
    )

    # 创建分词器
    print("\n" + "=" * 60)
    print("创建分词器")
    print("=" * 60)

    tokenizer = ChineseTokenizer(
        max_vocab_size=CONFIG['max_vocab_size'],
        use_word=CONFIG['use_word']
    )
    tokenizer.build_vocab(texts, min_freq=CONFIG['min_freq'])

    # 保存分词器
    tokenizer_path = os.path.join(CONFIG['save_dir'], 'tokenizer.pkl')
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    tokenizer.save(tokenizer_path)

    # 创建数据加载器
    print("\n" + "=" * 60)
    print("创建数据加载器")
    print("=" * 60)

    from sklearn.model_selection import train_test_split

    # 分割数据
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")

    # 创建数据集
    train_dataset = SentimentDataset(
        train_texts, train_labels, tokenizer, max_length=CONFIG['max_length']
    )
    val_dataset = SentimentDataset(
        val_texts, val_labels, tokenizer, max_length=CONFIG['max_length']
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0
    )

    # 创建模型
    print("\n" + "=" * 60)
    print("创建模型")
    print("=" * 60)

    model_config = {
        'd_model': CONFIG['d_model'],
        'num_heads': CONFIG['num_heads'],
        'num_layers': CONFIG['num_layers'],
        'd_ff': CONFIG['d_ff'],
        'dropout': CONFIG['dropout'],
        'max_len': CONFIG['max_length']
    }

    model = create_model(
        vocab_size=tokenizer.vocab_size,
        num_classes=3,
        config=model_config
    )
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建优化器和损失函数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # 计算类别权重以处理类别不平衡
    from collections import Counter
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = 3

    # 计算权重：样本数少的类别权重更高
    class_weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = total_samples / (num_classes * count)
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"\n类别权重 (处理不平衡): {class_weights.tolist()}")

    # 使用标签平滑的损失函数，帮助泛化
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=CONFIG.get('label_smoothing', 0.1))

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=CONFIG['save_dir'],
        num_classes=3
    )

    # 训练模型
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        early_stopping_patience=CONFIG['early_stopping_patience']
    )

    # 绘制训练曲线
    trainer.plot_history(os.path.join(CONFIG['save_dir'], 'training_history.png'))

    # 打印分类报告
    trainer.print_classification_report()

    # 保存配置
    config_path = os.path.join(CONFIG['save_dir'], 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)
    print(f"\n配置已保存到 {config_path}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()