"""
中文分词器和数据处理模块
支持字符级和词级分词
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import os
import re
from collections import Counter


class ChineseTokenizer:
    """
    中文分词器
    支持字符级分词和基于 jieba 的词级分词
    """
    
    # 特殊 token
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    CLS_TOKEN = '<CLS>'
    SEP_TOKEN = '<SEP>'
    
    def __init__(self, vocab=None, max_vocab_size=30000, use_word=False):
        """
        Args:
            vocab: 预定义的词汇表（字典）
            max_vocab_size: 最大词汇表大小
            use_word: 是否使用词级分词（需要 jieba）
        """
        self.max_vocab_size = max_vocab_size
        self.use_word = use_word
        self.vocab = vocab
        self.inverse_vocab = None
        
        if vocab is not None:
            self._build_inverse_vocab()
        
        # 特殊 token 的 ID
        self.pad_id = 0
        self.unk_id = 1
        self.cls_id = 2
        self.sep_id = 3
        
        # 尝试导入 jieba
        if use_word:
            try:
                import jieba
                self.jieba = jieba
            except ImportError:
                print("警告: 未安装 jieba，将使用字符级分词")
                self.use_word = False
                self.jieba = None
        else:
            self.jieba = None
    
    def _tokenize_text(self, text):
        """
        对单个文本进行分词
        """
        # 清理文本
        text = self._clean_text(text)
        
        if self.use_word and self.jieba:
            # 词级分词
            tokens = list(self.jieba.cut(text))
        else:
            # 字符级分词
            tokens = list(text)
        
        return tokens
    
    def _clean_text(self, text):
        """
        清理文本 - 保留情感相关标点符号
        """
        # 移除多余空白
        text = re.sub(r'\s+', '', text)
        # 保留中文、英文、数字和情感标点（！！?！！！??等）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）【】!？?]', '', text)
        return text
    
    def build_vocab(self, texts, min_freq=2):
        """
        从文本列表构建词汇表
        
        Args:
            texts: 文本列表
            min_freq: 最小词频阈值
        """
        print("正在构建词汇表...")
        
        # 统计词频
        counter = Counter()
        for text in texts:
            tokens = self._tokenize_text(text)
            counter.update(tokens)
        
        # 添加特殊 token
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN]
        
        # 选择高频词
        vocab_tokens = []
        for token, freq in counter.most_common(self.max_vocab_size - len(special_tokens)):
            if freq >= min_freq:
                vocab_tokens.append(token)
        
        # 构建词汇表
        self.vocab = {}
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
        
        for i, token in enumerate(vocab_tokens, start=len(special_tokens)):
            self.vocab[token] = i
        
        self._build_inverse_vocab()
        
        print(f"词汇表构建完成，共 {len(self.vocab)} 个token")
        
        return self.vocab
    
    def _build_inverse_vocab(self):
        """
        构建反向词汇表
        """
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text, max_length=128, add_special_tokens=True):
        """
        将文本编码为 token ID 序列
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            add_special_tokens: 是否添加特殊 token
        
        Returns:
            input_ids: token ID 列表
            attention_mask: 注意力掩码列表
        """
        if self.vocab is None:
            raise ValueError("词汇表未初始化，请先调用 build_vocab 或加载预训练词汇表")
        
        # 分词
        tokens = self._tokenize_text(text)
        
        # 转换为 ID
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.cls_id)
        
        for token in tokens:
            token_id = self.vocab.get(token, self.unk_id)
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.sep_id)
        
        # 截断
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.sep_id]
        
        # 创建注意力掩码
        attention_mask = [1] * len(token_ids)
        
        # 填充
        padding_length = max_length - len(token_ids)
        token_ids = token_ids + [self.pad_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        
        return token_ids, attention_mask
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        将 token ID 序列解码为文本
        
        Args:
            token_ids: token ID 列表
            skip_special_tokens: 是否跳过特殊 token
        """
        if self.inverse_vocab is None:
            raise ValueError("反向词汇表未初始化")
        
        tokens = []
        special_ids = {self.pad_id, self.unk_id, self.cls_id, self.sep_id}
        
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            token = self.inverse_vocab.get(token_id, self.UNK_TOKEN)
            tokens.append(token)
        
        return ''.join(tokens)
    
    @property
    def vocab_size(self):
        """返回词汇表大小"""
        return len(self.vocab) if self.vocab else 0
    
    def save(self, path):
        """
        保存分词器
        """
        data = {
            'vocab': self.vocab,
            'max_vocab_size': self.max_vocab_size,
            'use_word': self.use_word,
            'pad_id': self.pad_id,
            'unk_id': self.unk_id,
            'cls_id': self.cls_id,
            'sep_id': self.sep_id
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"分词器已保存到 {path}")
    
    @classmethod
    def load(cls, path):
        """
        加载分词器
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(
            vocab=data['vocab'],
            max_vocab_size=data['max_vocab_size'],
            use_word=data['use_word']
        )
        
        tokenizer.pad_id = data['pad_id']
        tokenizer.unk_id = data['unk_id']
        tokenizer.cls_id = data['cls_id']
        tokenizer.sep_id = data['sep_id']
        
        print(f"分词器已从 {path} 加载")
        
        return tokenizer


class SentimentDataset(Dataset):
    """
    情感分析数据集
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器实例
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"数据集创建完成，共 {len(self.texts)} 条数据")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        input_ids, attention_mask = self.tokenizer.encode(
            text, 
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def _normalize_labels_to_int(labels):
    """
    统一为 0/1/2 的 int，避免 pandas 读成 float/str 时与 CrossEntropy 语义不一致。
    """
    out = []
    for y in labels:
        if y is None or (isinstance(y, float) and np.isnan(y)):
            raise ValueError(f"标签存在空值: {y!r}")
        try:
            v = int(float(str(y).strip()))
        except (ValueError, TypeError) as e:
            raise ValueError(f"标签无法转为整数: {y!r}") from e
        if v not in (0, 1, 2):
            raise ValueError(f"标签必须为 0、1 或 2，当前: {v} (来自 {y!r})")
        out.append(v)
    return out


def load_data_from_csv(csv_path, text_column='sentence', label_column='label'):
    """
    从 CSV 文件加载数据
    
    Args:
        csv_path: CSV 文件路径
        text_column: 文本列名
        label_column: 标签列名
    
    Returns:
        texts: 文本列表
        labels: 标签列表
    """
    print(f"正在从 {csv_path} 加载数据...")
    
    # 尝试不同的编码
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"使用编码 {encoding} 成功加载数据")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise RuntimeError(f"无法加载 CSV 文件 {csv_path}，尝试了多种编码均失败")
    
    # 检查列名
    if text_column not in df.columns:
        raise KeyError(f"CSV 文件中不存在 '{text_column}' 列，可用列名: {list(df.columns)}")
    if label_column not in df.columns:
        raise KeyError(f"CSV 文件中不存在 '{label_column}' 列，可用列名: {list(df.columns)}")
    
    texts = df[text_column].tolist()
    labels = _normalize_labels_to_int(df[label_column].tolist())
    
    print(f"数据加载完成，共 {len(texts)} 条")
    
    # 打印标签分布
    label_counts = Counter(labels)
    print(f"标签分布: {dict(label_counts)}")
    
    return texts, labels


def create_dataloaders(
    texts, 
    labels, 
    tokenizer, 
    batch_size=16, 
    max_length=128, 
    test_size=0.2,
    random_state=42
):
    """
    创建训练和验证数据加载器
    
    Args:
        texts: 文本列表
        labels: 标签列表
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        test_size: 验证集比例
        random_state: 随机种子
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    from sklearn.model_selection import train_test_split
    
    # 分割数据集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    
    # 创建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试分词器
    print("=" * 60)
    print("测试中文分词器")
    print("=" * 60)
    
    # 测试文本
    test_texts = [
        "今天天气真好，心情很愉快",
        "这个产品质量太差了，我很失望",
        "这个产品还可以，没什么特别的",
        "我真的很无语，不知道该说什么",
        "太棒了，终于完成了这个项目！"
    ]
    
    # 创建分词器
    tokenizer = ChineseTokenizer(max_vocab_size=1000, use_word=False)
    
    # 构建词汇表
    tokenizer.build_vocab(test_texts, min_freq=1)
    
    print(f"\n词汇表大小: {tokenizer.vocab_size}")
    
    # 测试编码
    print("\n测试编码:")
    for text in test_texts[:2]:
        input_ids, attention_mask = tokenizer.encode(text, max_length=20)
        print(f"\n原文: {text}")
        print(f"Token IDs: {input_ids}")
        print(f"Attention Mask: {attention_mask}")
        
        # 解码
        decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"解码: {decoded}")
    
    # 测试数据集
    print("\n" + "=" * 60)
    print("测试数据集")
    print("=" * 60)
    
    labels = [1, 0, 2, 0, 1]  # 1: 正面, 0: 负面, 2: 中性
    
    dataset = SentimentDataset(test_texts, labels, tokenizer, max_length=32)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    sample = dataset[0]
    print(f"\n样本 0:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  label: {sample['label']}")
    
    # 测试数据加载器
    print("\n" + "=" * 60)
    print("测试数据加载器")
    print("=" * 60)
    
    # 使用更多样本进行测试
    extended_texts = test_texts * 3  # 复制样本
    extended_labels = labels * 3
    
    train_loader, val_loader = create_dataloaders(
        extended_texts, extended_labels, tokenizer, batch_size=2, max_length=32, test_size=0.3
    )
    
    print(f"\n训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 获取一个批次
    batch = next(iter(train_loader))
    print(f"\n批次数据:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['label'].shape}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
