import torch
import torch.nn as nn
import json
import os
import sys
import time
from typing import List, Dict, Tuple, Optional
import argparse


# ==================== 模型定义（与训练代码一致） ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, q_seq_len, _ = q.size()
        _, k_seq_len, _ = k.size()

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.contiguous().view(batch_size, q_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.contiguous().view(batch_size, k_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.contiguous().view(batch_size, k_seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_model)

        return self.w_o(output)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, n_layers=2,
                 n_heads=4, d_ff=512, dropout=0.1, max_len=100):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_pos_encoding = PositionalEncoding(d_model, max_len)

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_ids, src_mask=None):
        src_emb = self.src_embedding(src_ids) * math.sqrt(self.d_model)
        src_emb = self.src_pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)

        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        return memory

    def decode(self, tgt_ids, memory, src_mask=None, tgt_mask=None):
        tgt_emb = self.tgt_embedding(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        x = tgt_emb
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return x

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        memory = self.encode(src_ids, src_mask)
        decoder_output = self.decode(tgt_ids, memory, src_mask, tgt_mask)
        output = self.output_layer(decoder_output)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# ==================== 推理接口类 ====================

class TranslationInterface:
    """中英翻译推理接口"""

    def __init__(self, model_path='checkpoints/model_epoch_3.pth',
                 vocab_path='checkpoints/vocab.json',
                 device=None):
        """
        初始化翻译接口

        参数:
            model_path: 模型权重文件路径
            vocab_path: 词汇表文件路径
            device: 指定设备 (cpu/cuda)
        """
        import math
        import torch.nn.functional as F
        global math, F

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")
        print(f"加载模型: {model_path}")

        # 加载词汇表
        self.load_vocab(vocab_path)

        # 创建模型
        self.model = Transformer(
            src_vocab_size=self.src_tokenizer.vocab_size,
            tgt_vocab_size=self.tgt_tokenizer.vocab_size,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            max_len=100
        ).to(self.device)

        # 加载模型权重
        self.load_model(model_path)

        # 设置模型为评估模式
        self.model.eval()

        print(f"模型加载完成!")
        print(f"中文词汇表大小: {self.src_tokenizer.vocab_size}")
        print(f"英文词汇表大小: {self.tgt_tokenizer.vocab_size}")

    def load_vocab(self, vocab_path):
        """加载词汇表"""
        try:
            # 尝试从文件加载词汇表
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            # 创建中文分词器
            self.src_tokenizer = SimpleTokenizer(lang='chinese')
            self.src_tokenizer.token2id = vocab_data['src_token2id']
            self.src_tokenizer.id2token = {int(k): v for k, v in vocab_data['src_id2token'].items()}
            self.src_tokenizer.vocab_size = len(self.src_tokenizer.token2id)

            # 创建英文分词器
            self.tgt_tokenizer = SimpleTokenizer(lang='english')
            self.tgt_tokenizer.token2id = vocab_data['tgt_token2id']
            self.tgt_tokenizer.id2token = {int(k): v for k, v in vocab_data['tgt_id2token'].items()}
            self.tgt_tokenizer.vocab_size = len(self.tgt_tokenizer.token2id)

            print("从文件加载词汇表成功")

        except Exception as e:
            print(f"加载词汇表失败: {e}")
            print("创建默认词汇表...")
            self.create_default_vocab()

    def create_default_vocab(self):
        """创建默认词汇表（用于演示）"""
        # 中文分词器
        self.src_tokenizer = SimpleTokenizer(lang='chinese')
        self.src_tokenizer.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

        # 创建基础词汇表
        self.src_tokenizer.token2id = {
            '<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3
        }
        for i, char in enumerate("我今天天气很好喜欢学习人工智能"):
            self.src_tokenizer.token2id[char] = i + 4

        self.src_tokenizer.id2token = {v: k for k, v in self.src_tokenizer.token2id.items()}
        self.src_tokenizer.vocab_size = len(self.src_tokenizer.token2id)

        # 英文分词器
        self.tgt_tokenizer = SimpleTokenizer(lang='english')
        self.tgt_tokenizer.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

        self.tgt_tokenizer.token2id = {
            '<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3
        }
        english_words = ["the", "weather", "is", "very", "good", "today",
                         "i", "like", "learning", "artificial", "intelligence",
                         "hello", "world", "how", "are", "you"]
        for i, word in enumerate(english_words):
            self.tgt_tokenizer.token2id[word] = i + 4

        self.tgt_tokenizer.id2token = {v: k for k, v in self.tgt_tokenizer.token2id.items()}
        self.tgt_tokenizer.vocab_size = len(self.tgt_tokenizer.token2id)

        print("创建默认词汇表完成")

    def load_model(self, model_path):
        """加载模型权重"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型权重加载成功")
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            print("使用随机初始化的模型")

    def preprocess_text(self, text, is_source=True):
        """
        预处理文本，转换为token ids

        参数:
            text: 输入文本
            is_source: 是否为源语言（中文）

        返回:
            token_ids: token id列表
            tokens: token列表
        """
        if is_source:
            tokenizer = self.src_tokenizer
        else:
            tokenizer = self.tgt_tokenizer

        # 分词
        tokens = tokenizer.tokenize(text)

        # 添加特殊标记
        tokens = ['<sos>'] + tokens + ['<eos>']

        # 转换为id
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        return token_ids, tokens

    def translate(self, chinese_text, max_length=50, beam_size=1):
        """
        翻译中文文本到英文

        参数:
            chinese_text: 中文文本
            max_length: 最大生成长度
            beam_size: beam search的大小（1表示贪心搜索）

        返回:
            english_text: 英文翻译
            tokens: 翻译的token列表
            inference_time: 推理时间（秒）
        """
        start_time = time.time()

        # 预处理输入
        src_ids, src_tokens = self.preprocess_text(chinese_text, is_source=True)

        # 转换为tensor
        src_ids = torch.tensor([src_ids], dtype=torch.long).to(self.device)
        src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)

        # 生成翻译
        if beam_size == 1:
            # 贪心搜索
            english_text, english_tokens = self.greedy_decode(src_ids, src_mask, max_length)
        else:
            # Beam Search
            english_text, english_tokens = self.beam_search_decode(src_ids, src_mask, max_length, beam_size)

        inference_time = time.time() - start_time

        return english_text, english_tokens, inference_time

    def greedy_decode(self, src_ids, src_mask, max_length):
        """贪心搜索解码"""
        # 初始化解码序列
        start_token = 1  # <sos>
        end_token = 2  # <eos>

        generated = torch.tensor([[start_token]], dtype=torch.long).to(self.device)

        # 逐步生成
        for step in range(max_length):
            # 创建目标掩码
            tgt_mask = self.model.generate_square_subsequent_mask(generated.size(1)).to(self.device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

            # 前向传播
            with torch.no_grad():
                output = self.model(src_ids, generated, src_mask, tgt_mask)

            # 选择概率最高的token
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # 如果生成了结束符，停止生成
            if next_token.item() == end_token:
                break

        # 转换为文本
        english_tokens = []
        for token_id in generated[0][1:]:  # 跳过<sos>
            if token_id.item() == end_token:
                break
            if token_id.item() != 0:
                token = self.tgt_tokenizer.id2token.get(token_id.item(), '<unk>')
                if token not in ['<sos>', '<eos>', '<pad>', '<unk>']:
                    english_tokens.append(token)

        english_text = ' '.join(english_tokens)

        return english_text, english_tokens

    def beam_search_decode(self, src_ids, src_mask, max_length, beam_size):
        """Beam Search解码（简化版）"""
        start_token = 1
        end_token = 2

        # 初始化beam
        beams = [{'tokens': [start_token], 'score': 0.0}]

        for step in range(max_length):
            new_beams = []

            for beam in beams:
                if beam['tokens'][-1] == end_token:
                    # 如果已经生成结束符，保持原样
                    new_beams.append(beam)
                    continue

                # 将当前beam转换为tensor
                tokens_tensor = torch.tensor([beam['tokens']], dtype=torch.long).to(self.device)

                # 创建目标掩码
                tgt_mask = self.model.generate_square_subsequent_mask(tokens_tensor.size(1)).to(self.device)
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

                # 前向传播
                with torch.no_grad():
                    output = self.model(src_ids, tokens_tensor, src_mask, tgt_mask)

                # 获取最后一个token的概率分布
                log_probs = F.log_softmax(output[:, -1, :], dim=-1)

                # 选择top-k个候选
                topk_probs, topk_indices = log_probs.topk(beam_size, dim=-1)

                for i in range(beam_size):
                    new_token = topk_indices[0, i].item()
                    new_score = beam['score'] + topk_probs[0, i].item()
                    new_tokens = beam['tokens'] + [new_token]

                    new_beams.append({
                        'tokens': new_tokens,
                        'score': new_score
                    })

            # 选择分数最高的beam_size个beam
            new_beams.sort(key=lambda x: x['score'], reverse=True)
            beams = new_beams[:beam_size]

            # 如果所有beam都以结束符结尾，停止生成
            if all(beam['tokens'][-1] == end_token for beam in beams):
                break

        # 选择分数最高的beam
        best_beam = beams[0]

        # 转换为文本
        english_tokens = []
        for token_id in best_beam['tokens'][1:]:  # 跳过<sos>
            if token_id == end_token:
                break
            if token_id != 0:
                token = self.tgt_tokenizer.id2token.get(token_id, '<unk>')
                if token not in ['<sos>', '<eos>', '<pad>', '<unk>']:
                    english_tokens.append(token)

        english_text = ' '.join(english_tokens)

        return english_text, english_tokens

    def batch_translate(self, chinese_texts, max_length=50, beam_size=1):
        """
        批量翻译

        参数:
            chinese_texts: 中文文本列表
            max_length: 最大生成长度
            beam_size: beam search大小

        返回:
            results: 翻译结果列表，每个元素是(原文, 译文, 推理时间)的元组
        """
        results = []

        for text in chinese_texts:
            english_text, tokens, inference_time = self.translate(
                text, max_length=max_length, beam_size=beam_size
            )
            results.append({
                'source': text,
                'translation': english_text,
                'tokens': tokens,
                'time': inference_time
            })

        return results

    def interactive_mode(self):
        """交互式翻译模式"""
        print("\n" + "=" * 60)
        print("Transformer中英翻译系统 - 交互模式")
        print("=" * 60)
        print("输入 'quit' 或 'exit' 退出程序")
        print("输入 'help' 查看帮助")
        print("-" * 60)

        while True:
            try:
                # 获取用户输入
                user_input = input("\n请输入中文: ").strip()

                # 退出命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("感谢使用，再见！")
                    break

                # 帮助命令
                if user_input.lower() in ['help', 'h']:
                    self.show_help()
                    continue

                # 空输入
                if not user_input:
                    print("输入不能为空，请重新输入")
                    continue

                # 开始翻译
                print(f"\n翻译中...")
                english_text, tokens, inference_time = self.translate(user_input)

                # 显示结果
                print(f"\n原文: {user_input}")
                print(f"翻译: {english_text}")
                print(f"推理时间: {inference_time:.3f}秒")
                if len(tokens) < 10:  # 只显示较短的token列表
                    print(f"Tokens: {tokens}")

            except KeyboardInterrupt:
                print("\n\n检测到中断信号，退出程序")
                break
            except Exception as e:
                print(f"\n翻译出错: {e}")

    def show_help(self):
        """显示帮助信息"""
        print("\n" + "=" * 60)
        print("帮助信息")
        print("=" * 60)
        print("可用命令:")
        print("  quit, exit, q - 退出程序")
        print("  help, h - 显示帮助信息")
        print("\n使用示例:")
        print("  输入: 今天天气很好")
        print("  输出: weather good today")
        print("\n注意事项:")
        print("  1. 输入应为中文文本")
        print("  2. 句子不宜过长（建议不超过30字）")
        print("  3. 模型对常见短语翻译效果较好")
        print("=" * 60)


# ==================== 简化的分词器类 ====================

class SimpleTokenizer:
    """简化的分词器类"""

    def __init__(self, lang='chinese'):
        self.lang = lang
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.token2id = {}
        self.id2token = {}
        self.vocab_size = 0

    def tokenize(self, text):
        if self.lang == 'chinese':
            return list(text)
        else:
            tokens = text.lower().split()
            processed_tokens = []
            for token in tokens:
                if len(token) > 0:
                    processed_tokens.append(token)
            return processed_tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.token2id.get(token, self.token2id.get('<unk>', 3)) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id2token.get(id, '<unk>') for id in ids]


# ==================== 主函数 ====================

def main():
    """主函数：解析命令行参数并运行翻译接口"""
    parser = argparse.ArgumentParser(description='Transformer中英翻译系统推理接口')

    # 命令行参数
    parser.add_argument('--model', type=str, default='checkpoints/model_epoch_3.pth',
                        help='模型权重文件路径')
    parser.add_argument('--vocab', type=str, default='checkpoints/vocab.json',
                        help='词汇表文件路径')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='指定运行设备 (cpu/cuda)')
    parser.add_argument('--text', type=str, default=None,
                        help='要翻译的中文文本（单句）')
    parser.add_argument('--file', type=str, default=None,
                        help='包含中文文本的文件（每行一句）')
    parser.add_argument('--max-length', type=int, default=50,
                        help='最大生成长度')
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam Search的大小（1表示贪心搜索）')
    parser.add_argument('--interactive', action='store_true',
                        help='进入交互模式')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果文件路径')

    args = parser.parse_args()

    # 创建翻译接口
    try:
        translator = TranslationInterface(
            model_path=args.model,
            vocab_path=args.vocab,
            device=args.device
        )
    except Exception as e:
        print(f"初始化翻译接口失败: {e}")
        print("请确保模型文件和词汇表文件存在")
        print("如果文件不存在，将使用默认模型进行演示")

        # 使用默认配置创建接口
        translator = TranslationInterface(
            model_path=None,
            vocab_path=None,
            device=args.device
        )

    # 根据参数选择运行模式
    if args.interactive:
        # 交互模式
        translator.interactive_mode()

    elif args.text:
        # 单句翻译模式
        print(f"\n翻译文本: {args.text}")
        english_text, tokens, inference_time = translator.translate(
            args.text,
            max_length=args.max_length,
            beam_size=args.beam_size
        )

        print(f"\n原文: {args.text}")
        print(f"翻译: {english_text}")
        print(f"推理时间: {inference_time:.3f}秒")
        print(f"Tokens: {tokens}")

        # 保存结果到文件
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"原文: {args.text}\n")
                f.write(f"翻译: {english_text}\n")
                f.write(f"推理时间: {inference_time:.3f}秒\n")
            print(f"结果已保存到: {args.output}")

    elif args.file:
        # 文件翻译模式
        print(f"\n从文件读取文本: {args.file}")

        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

            print(f"读取到 {len(texts)} 条文本")

            # 批量翻译
            results = translator.batch_translate(
                texts,
                max_length=args.max_length,
                beam_size=args.beam_size
            )

            # 显示结果
            print("\n" + "=" * 60)
            print("翻译结果:")
            print("=" * 60)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. 原文: {result['source']}")
                print(f"   翻译: {result['translation']}")
                print(f"   时间: {result['time']:.3f}秒")

            # 保存结果到文件
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(f"原文: {result['source']}\n")
                        f.write(f"翻译: {result['translation']}\n")
                        f.write(f"时间: {result['time']:.3f}秒\n")
                        f.write("-" * 40 + "\n")
                print(f"\n所有结果已保存到: {args.output}")

            # 统计信息
            total_time = sum(r['time'] for r in results)
            avg_time = total_time / len(results) if results else 0
            print(f"\n统计信息:")
            print(f"  总翻译数量: {len(results)}")
            print(f"  总推理时间: {total_time:.3f}秒")
            print(f"  平均推理时间: {avg_time:.3f}秒/句")

        except Exception as e:
            print(f"读取文件失败: {e}")

    else:
        # 默认：显示帮助并进入交互模式
        print("未指定翻译文本或文件，进入交互模式")
        print("使用 --help 查看所有选项")
        print("-" * 60)
        translator.interactive_mode()


# ==================== 快捷使用示例 ====================

def quick_translate(text, model_path=None, device=None):
    """
    快速翻译函数（用于在代码中直接调用）

    示例:
        result = quick_translate("今天天气很好")
        print(result)
    """
    # 创建翻译接口
    translator = TranslationInterface(
        model_path=model_path,
        vocab_path=None,
        device=device
    )

    # 执行翻译
    english_text, tokens, inference_time = translator.translate(text)

    return {
        'source': text,
        'translation': english_text,
        'tokens': tokens,
        'time': inference_time
    }


# ==================== 如果作为脚本运行 ====================

if __name__ == "__main__":
    # 检查必需的库
    try:
        import torch
        import torch.nn.functional as F
        import math
    except ImportError as e:
        print(f"缺少必需的库: {e}")
        print("请安装: pip install torch")
        sys.exit(1)

    # 运行主函数
    main()