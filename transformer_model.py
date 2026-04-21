"""
纯 Transformer 架构的中文情感分析模型
从零实现 Transformer Encoder 用于文本分类任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    位置编码层
    使用正弦和余弦函数生成固定位置编码
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加 batch 维度
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，shape: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q, K, V 的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: 输入张量，shape: (batch_size, seq_len, d_model)
            mask: 注意力掩码，shape: (batch_size, seq_len)
            return_attention: 是否返回注意力权重
        Returns:
            输出张量，shape: (batch_size, seq_len, d_model)
            如果 return_attention=True，还返回注意力权重
        """
        batch_size, seq_len, _ = x.size()
        
        # 线性变换得到 Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在 shape: (batch_size, num_heads, seq_len, head_dim)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        
        # 应用掩码（用于padding）
        if mask is not None:
            # mask shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)
        
        # 保存原始注意力权重用于返回（不应用dropout）
        raw_attention_weights = attention_weights
        
        # 应用dropout（仅用于计算输出）
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        # output shape: (batch_size, num_heads, seq_len, head_dim)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出线性变换
        output = self.W_o(output)
        
        if return_attention:
            return output, raw_attention_weights
        return output


class FeedForward(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，shape: (batch_size, seq_len, d_model)
        """
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层
    包含多头自注意力和前馈网络
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, return_attention=False):
        # 多头自注意力 + 残差连接 + LayerNorm
        if return_attention:
            attn_output, attention_weights = self.self_attention(x, mask, return_attention=True)
        else:
            attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        if return_attention:
            return x, attention_weights
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    堆叠多个编码器层
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None, return_attention=False):
        all_attention_weights = []
        for layer in self.layers:
            if return_attention:
                x, attention_weights = layer(x, mask, return_attention=True)
                all_attention_weights.append(attention_weights)
            else:
                x = layer(x, mask)
        
        if return_attention:
            return x, all_attention_weights
        return x


class SentimentTransformer(nn.Module):
    """
    基于 Transformer 的情感分析模型
    """
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        num_classes=3,
        max_len=512,
        dropout=0.1,
        pad_idx=0
    ):
        super(SentimentTransformer, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer 编码器
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)

        # 池化后的LayerNorm，稳定分类器输入
        self.pool_norm = nn.LayerNorm(d_model)

        # 分类头 - 更稳定的结构
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型参数 - 使用更稳定的初始化"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 分类器使用更小的初始化
                if 'classifier' in name:
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # 降低embedding初始化标准差
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, return_attention=False):
        """
        Args:
            input_ids: 输入token ID，shape: (batch_size, seq_len)
            attention_mask: 注意力掩码，shape: (batch_size, seq_len)
            return_attention: 是否返回注意力权重
        Returns:
            logits: 分类 logits，shape: (batch_size, num_classes)
            如果 return_attention=True，还返回注意力权重列表
        """
        # 词嵌入
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.positional_encoding(x)
        
        # Transformer 编码器
        if return_attention:
            x, attention_weights = self.encoder(x, attention_mask, return_attention=True)
        else:
            x = self.encoder(x, attention_mask)
        
        # 池化：使用第一个 token 或平均池化
        # 这里使用平均池化（排除 padding）
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)

        # 池化后的LayerNorm，稳定分类器输入
        x = self.pool_norm(x)

        # 分类
        logits = self.classifier(x)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def predict(self, input_ids, attention_mask=None):
        """
        预测类别
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities


def create_model(vocab_size, num_classes=3, config=None):
    """
    创建情感分析 Transformer 模型的便捷函数
    
    Args:
        vocab_size: 词汇表大小
        num_classes: 类别数量（默认3：正面、负面、中性）
        config: 可选的配置字典
    
    Returns:
        SentimentTransformer 模型实例
    """
    default_config = {
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,
        'd_ff': 512,
        'max_len': 512,
        'dropout': 0.1,
        'pad_idx': 0
    }
    
    if config:
        default_config.update(config)
    
    model = SentimentTransformer(
        vocab_size=vocab_size,
        num_classes=num_classes,
        **default_config
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("测试 Transformer 情感分析模型")
    print("=" * 60)
    
    # 模型参数
    vocab_size = 10000
    batch_size = 4
    seq_len = 32
    
    # 创建模型
    model = create_model(vocab_size=vocab_size, num_classes=3)
    
    # 打印模型结构
    print(f"\n模型结构:")
    print(model)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试前向传播
    print(f"\n测试前向传播:")
    print(f"  输入形状: (batch_size={batch_size}, seq_len={seq_len})")
    
    # 随机输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, seq_len//2:] = 0  # 模拟 padding
    
    # 前向传播
    output = model(input_ids, attention_mask)
    print(f"  输出形状: {output.shape}")
    
    # 预测
    predictions, probabilities = model.predict(input_ids, attention_mask)
    print(f"  预测形状: {predictions.shape}")
    print(f"  概率形状: {probabilities.shape}")
    
    print("\n" + "=" * 60)
    print("模型测试完成！")
    print("=" * 60)
