"""
Transformer 情感分析模型推理接口
提供简单的 API 用于情感预测
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Union, Optional
import json

from transformer_model import create_model
from tokenizer import ChineseTokenizer

# 与训练保存的 config.json 对齐时，只采纳这些键构建网络（忽略 data_path、batch_size 等）
_MODEL_ARCH_KEYS = frozenset(
    {'d_model', 'num_heads', 'num_layers', 'd_ff', 'dropout', 'max_length', 'max_len'}
)


class SentimentPredictor:
    """
    情感分析预测器
    """
    
    # 情感标签映射
    LABEL_MAP = {
        0: {'name': '负面', 'name_en': 'Negative', 'emoji': '😞', 'color': '#FF6B6B'},
        1: {'name': '正面', 'name_en': 'Positive', 'emoji': '😊', 'color': '#4ECDC4'},
        2: {'name': '中性', 'name_en': 'Neutral', 'emoji': '😐', 'color': '#95E1D3'}
    }
    
    def __init__(
        self,
        model_dir: str = 'transformer_checkpoints',
        model_file: str = 'best_model.pt',
        device: Optional[str] = None
    ):
        """
        初始化预测器
        
        Args:
            model_dir: 模型目录
            model_file: 模型文件名
            device: 设备 ('cuda', 'cpu' 或 None 自动选择)
        """
        print("开始初始化预测器...")
        self.model_dir = model_dir
        self.model_file = model_file
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")
        
        # 加载配置
        print("加载配置...")
        self.config = self._load_config()
        print(f"配置: {self.config}")
        
        # 加载分词器
        print("加载分词器...")
        self.tokenizer = self._load_tokenizer()
        print(f"分词器词汇表大小: {self.tokenizer.vocab_size}")
        
        # 加载模型
        print("加载模型...")
        self.model = self._load_model()
        
        print(f"预测器已初始化，使用设备: {self.device}")
    
    def _load_config(self) -> Dict:
        """从 checkpoint 目录读取 config.json，并与当前训练默认结构对齐。"""
        defaults = {
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'd_ff': 512,
            'dropout': 0.15,
            'max_length': 128,
        }
        config_path = os.path.join(self.model_dir, 'config.json')
        merged = defaults.copy()
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            for k in _MODEL_ARCH_KEYS:
                if k in loaded:
                    merged[k] = loaded[k]
            print(f"已从 {config_path} 读取模型结构相关配置")
        else:
            print(f"未找到 {config_path}，使用默认模型结构（与 train_transformer 一致）")
        return merged
    
    def _load_tokenizer(self) -> ChineseTokenizer:
        """加载分词器"""
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pkl')
        
        if os.path.exists(tokenizer_path):
            tokenizer = ChineseTokenizer.load(tokenizer_path)
        else:
            raise FileNotFoundError(f"分词器文件未找到: {tokenizer_path}")
        
        return tokenizer
    
    def _load_model(self) -> torch.nn.Module:
        """加载模型"""
        max_len = self.config.get('max_len', self.config.get('max_length', 128))

        # 加载权重
        model_path = os.path.join(self.model_dir, self.model_file)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        model_state = checkpoint['model_state_dict']

        # 从 checkpoint 权重中推断真实的模型结构参数，避免 config.json 不一致
        real_d_model = model_state['embedding.weight'].shape[1]

        # 推断 num_layers：统计 encoder.layers.X 的层数
        layer_indices = set()
        for k in model_state:
            if k.startswith('encoder.layers.'):
                layer_indices.add(int(k.split('.')[2]))
        real_num_layers = len(layer_indices)

        # 推断 d_ff：从 feed_forward.linear1.weight 的输出维度
        ff_key = 'encoder.layers.0.feed_forward.linear1.weight'
        real_d_ff = model_state[ff_key].shape[0] if ff_key in model_state else self.config.get('d_ff', 512)

        # num_heads 无法从权重推断，保持 config 中的值，但要确保能整除 d_model
        num_heads = self.config.get('num_heads', 8)
        if real_d_model % num_heads != 0:
            # 尝试常见的 num_heads 值
            for nh in [2, 4, 8]:
                if real_d_model % nh == 0:
                    num_heads = nh
                    break

        if log_changes := (real_d_model != self.config.get('d_model', 256)
                           or real_num_layers != self.config.get('num_layers', 4)
                           or real_d_ff != self.config.get('d_ff', 512)):
            print(f"WARNING: config.json params differ from actual weights, using weight values:")
            print(f"  d_model: {self.config.get('d_model')} -> {real_d_model}")
            print(f"  num_layers: {self.config.get('num_layers')} -> {real_num_layers}")
            print(f"  d_ff: {self.config.get('d_ff')} -> {real_d_ff}")

        # 优先使用checkpoint中的vocab_size
        vocab_size = checkpoint.get('vocab_size', self.tokenizer.vocab_size)
        print(f"使用vocab_size: {vocab_size} (来源: {'checkpoint' if 'vocab_size' in checkpoint else 'tokenizer'})")

        model_config = {
            'd_model': real_d_model,
            'num_heads': num_heads,
            'num_layers': real_num_layers,
            'd_ff': real_d_ff,
            'dropout': self.config.get('dropout', 0.15),
            'max_len': max_len,
            'pad_idx': 0,
        }

        print(f"使用的模型配置: {model_config}")

        model = create_model(
            vocab_size=vocab_size,
            num_classes=3,
            config=model_config
        )

        # 兼容性处理：旧模型没有pool_norm参数
        model_state_keys = set(model_state.keys())
        model_keys = set(model.state_dict().keys())

        # 检查是否缺少pool_norm参数
        missing_keys = model_keys - model_state_keys
        if missing_keys:
            print(f"检测到旧模型，缺少参数: {[k for k in missing_keys if 'pool_norm' in k]}")
            # 过滤掉pool_norm相关的键
            filtered_state = {k: v for k, v in model_state.items() if 'pool_norm' not in k}
            model.load_state_dict(filtered_state, strict=False)
            print(f"模型已从 {model_path} 加载（兼容模式，自动初始化缺失参数）")
        else:
            model.load_state_dict(model_state)
            print(f"模型已从 {model_path} 加载")

        if 'best_val_acc' in checkpoint:
            print(f"模型验证准确率: {checkpoint['best_val_acc']:.4f}")

        model = model.to(self.device)
        model.eval()

        return model
    
    def predict(self, text: str, return_all_scores: bool = True, return_attention: bool = False) -> Dict:
        """
        预测单个文本的情感
        
        Args:
            text: 输入文本
            return_all_scores: 是否返回所有类别的分数
            return_attention: 是否返回注意力权重
        
        Returns:
            预测结果字典
        """
        # 与训练时保持一致的文本预处理：移除文本开头的编号前缀
        import re
        text = re.sub(r'^\s*\d+[\.、,，]\s*', '', str(text))
        
        # 编码文本
        input_ids, attention_mask = self.tokenizer.encode(
            text,
            max_length=self.config.get('max_length', 128),
            add_special_tokens=True
        )
        
        # 转换为张量
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            if return_attention:
                logits, attention_weights = self.model(input_ids, attention_mask, return_attention=True)
            else:
                logits = self.model(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
        
        # 获取结果
        probs = probabilities[0].cpu().numpy()
        pred_label = int(probs.argmax())
        confidence = float(probs[pred_label])
        
        # 构建结果
        result = {
            'text': text,
            'label': pred_label,
            'label_name': self.LABEL_MAP[pred_label]['name'],
            'label_name_en': self.LABEL_MAP[pred_label]['name_en'],
            'emoji': self.LABEL_MAP[pred_label]['emoji'],
            'confidence': confidence,
            'confidence_percent': f"{confidence:.2%}"
        }
        
        if return_all_scores:
            result['all_scores'] = {
                self.LABEL_MAP[i]['name']: float(probs[i])
                for i in range(len(probs))
            }
            result['all_scores_percent'] = {
                self.LABEL_MAP[i]['name']: f"{float(probs[i]):.2%}"
                for i in range(len(probs))
            }
        
        if return_attention:
            # 获取tokens用于可视化
            tokens = self._get_tokens(input_ids[0], attention_mask[0])
            # 处理注意力权重
            attention_data = self._process_attention_weights(attention_weights, attention_mask[0])
            result['tokens'] = tokens
            result['attention_weights'] = attention_data
        
        return result
    
    def _get_tokens(self, input_ids, attention_mask):
        """将input_ids转换为tokens"""
        tokens = []
        for i, idx in enumerate(input_ids.cpu().numpy()):
            if attention_mask[i] == 0:  # padding
                break
            # 使用 inverse_vocab 获取 token
            token = self.tokenizer.inverse_vocab.get(int(idx), self.tokenizer.UNK_TOKEN)
            # 确保返回的是字符串而不是ID
            if not isinstance(token, str):
                token = str(token)
            tokens.append(token)
        return tokens
    
    def _process_attention_weights(self, attention_weights, attention_mask):
        """
        处理注意力权重，返回可用于可视化的数据
        
        Args:
            attention_weights: 列表，每个元素是一个层的注意力权重
                               shape: (batch_size, num_heads, seq_len, seq_len)
            attention_mask: 注意力掩码
        
        Returns:
            处理后的注意力数据
        """
        # 获取有效长度
        valid_len = attention_mask.sum().item()
        
        processed = []
        for layer_idx, layer_attention in enumerate(attention_weights):
            # layer_attention shape: (1, num_heads, seq_len, seq_len)
            layer_attn = layer_attention[0, :, :valid_len, :valid_len].cpu().numpy()
            
            # 平均所有头的注意力
            avg_attention = layer_attn.mean(axis=0)  # (seq_len, seq_len)
            
            processed.append({
                'layer': layer_idx,
                'attention': avg_attention.tolist(),
                'num_heads': layer_attn.shape[0]
            })
        
        return processed
    
    def predict_batch(self, texts: List[str], return_all_scores: bool = True) -> List[Dict]:
        """
        批量预测多个文本的情感
        
        Args:
            texts: 文本列表
            return_all_scores: 是否返回所有类别的分数
        
        Returns:
            预测结果列表
        """
        results = []
        
        for text in texts:
            result = self.predict(text, return_all_scores)
            results.append(result)
        
        return results
    
    def get_sentiment_distribution(self, texts: List[str]) -> Dict:
        """
        获取文本集合的情感分布统计
        
        Args:
            texts: 文本列表
        
        Returns:
            情感分布统计
        """
        results = self.predict_batch(texts, return_all_scores=False)
        
        # 统计各情感数量
        label_counts = {0: 0, 1: 0, 2: 0}
        total_confidence = {0: 0.0, 1: 0.0, 2: 0.0}
        
        for result in results:
            label = result['label']
            label_counts[label] += 1
            total_confidence[label] += result['confidence']
        
        total = len(texts)
        
        distribution = {
            'total': total,
            'counts': {
                self.LABEL_MAP[i]['name']: label_counts[i]
                for i in range(3)
            },
            'percentages': {
                self.LABEL_MAP[i]['name']: f"{label_counts[i] / total:.2%}"
                for i in range(3)
            },
            'avg_confidence': {
                self.LABEL_MAP[i]['name']: f"{total_confidence[i] / max(label_counts[i], 1):.2%}"
                for i in range(3)
            }
        }
        
        return distribution


def create_predictor(model_dir: str = 'transformer_checkpoints', model_file: str = 'best_model.pt') -> SentimentPredictor:
    """
    创建预测器的便捷函数
    
    Args:
        model_dir: 模型目录
        model_file: 模型文件名
    
    Returns:
        SentimentPredictor 实例
    """
    return SentimentPredictor(model_dir=model_dir, model_file=model_file)


# 命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer 情感分析预测')
    parser.add_argument('--model_dir', type=str, default='transformer_checkpoints',
                        help='模型目录')
    parser.add_argument('--model_file', type=str, default='best_model.pt',
                        help='模型文件名')
    parser.add_argument('--text', type=str, default=None,
                        help='要分析的文本')
    parser.add_argument('--interactive', action='store_true',
                        help='交互模式')
    
    args = parser.parse_args()
    
    # 创建预测器
    print("=" * 60)
    print("Transformer 情感分析预测器")
    print("=" * 60)
    
    try:
        predictor = create_predictor(args.model_dir, args.model_file)
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请先运行 train_transformer_web.py 训练模型")
        exit(1)
    
    print("\n" + "=" * 60)
    
    if args.interactive:
        # 交互模式
        print("交互模式 (输入 'quit' 退出)")
        print("-" * 40)
        
        while True:
            text = input("\n请输入文本: ").strip()
            
            if text.lower() == 'quit':
                print("再见！")
                break
            
            if not text:
                continue
            
            result = predictor.predict(text)
            
            print(f"\n结果:")
            print(f"  情感: {result['emoji']} {result['label_name']} ({result['label_name_en']})")
            print(f"  置信度: {result['confidence_percent']}")
            
            if 'all_scores_percent' in result:
                print(f"  各类别分数:")
                for name, score in result['all_scores_percent'].items():
                    print(f"    {name}: {score}")
    
    elif args.text:
        # 单次预测
        result = predictor.predict(args.text)
        
        print(f"\n输入文本: {result['text']}")
        print(f"\n结果:")
        print(f"  情感: {result['emoji']} {result['label_name']} ({result['label_name_en']})")
        print(f"  置信度: {result['confidence_percent']}")
        
        if 'all_scores_percent' in result:
            print(f"  各类别分数:")
            for name, score in result['all_scores_percent'].items():
                print(f"    {name}: {score}")
    
    else:
        # 演示模式
        print("演示模式")
        print("-" * 40)
        
        demo_texts = [
            "今天天气真好，心情很愉快",
            "这个产品质量太差了，我很失望",
            "这个产品还可以，没什么特别的",
            "太棒了，终于完成了这个项目！",
            "真的很无语，不知道该说什么"
        ]
        
        results = predictor.predict_batch(demo_texts)
        
        for result in results:
            print(f"\n文本: {result['text']}")
            print(f"  情感: {result['emoji']} {result['label_name']}")
            print(f"  置信度: {result['confidence_percent']}")
    
    print("\n" + "=" * 60)
