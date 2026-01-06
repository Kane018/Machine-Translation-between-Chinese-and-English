import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import math
import time
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 创建输出目录
os.makedirs('outputs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)


# ==================== 数据加载和预处理 ====================
class TranslationDataset(Dataset):
    def __init__(self, file_path, src_tokenizer, tgt_tokenizer, max_len=30):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    keys = list(item.keys())
                    if len(keys) >= 2:
                        src_text = item[keys[0]]
                        tgt_text = item[keys[1]]
                        if src_text and tgt_text:
                            self.data.append((src_text, tgt_text))
                except:
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        # 添加特殊标记
        src_tokens = ['<sos>'] + self.src_tokenizer.tokenize(src_text)[:self.max_len - 2] + ['<eos>']
        tgt_tokens = ['<sos>'] + self.tgt_tokenizer.tokenize(tgt_text)[:self.max_len - 2] + ['<eos>']

        # 转换为索引
        src_ids = self.src_tokenizer.convert_tokens_to_ids(src_tokens)
        tgt_ids = self.tgt_tokenizer.convert_tokens_to_ids(tgt_tokens)

        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


class SimpleTokenizer:
    def __init__(self, lang='chinese'):
        self.lang = lang
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.token2id = {}
        self.id2token = {}
        self.vocab_size = 0

    def build_vocab(self, texts, max_vocab_size=5000):
        token_freq = {}

        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1

        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        vocab_tokens = self.special_tokens + [token for token, _ in
                                              sorted_tokens[:max_vocab_size - len(self.special_tokens)]]

        for i, token in enumerate(vocab_tokens):
            self.token2id[token] = i
            self.id2token[i] = token

        self.vocab_size = len(vocab_tokens)

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
        return [self.token2id.get(token, self.token2id['<unk>']) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id2token.get(id, '<unk>') for id in ids]


# ==================== Transformer 模块 ====================
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

        # 线性变换
        q = self.w_q(q)  # [batch_size, q_seq_len, d_model]
        k = self.w_k(k)  # [batch_size, k_seq_len, d_model]
        v = self.w_v(v)  # [batch_size, k_seq_len, d_model]

        # 重塑为多头 - 修复维度问题
        q = q.contiguous().view(batch_size, q_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.contiguous().view(batch_size, k_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.contiguous().view(batch_size, k_seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # 扩展mask以匹配注意力分数的形状
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
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 交叉注意力 - 注意：x作为查询，memory作为键和值
        attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        # 前馈网络
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

        # 编码器
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_pos_encoding = PositionalEncoding(d_model, max_len)

        # 解码器
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len)

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        # 初始化参数
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


# ==================== 训练和评估函数 ====================
class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_losses = []
        self.valid_losses = []

    def create_mask(self, src_ids, tgt_ids):
        # 源序列掩码
        src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)

        # 目标序列掩码
        tgt_seq_len = tgt_ids.size(1)
        tgt_padding_mask = (tgt_ids != 0).unsqueeze(1).unsqueeze(3)
        tgt_subsequent_mask = self.model.generate_square_subsequent_mask(tgt_seq_len).to(self.device)
        tgt_mask = tgt_padding_mask & tgt_subsequent_mask.bool()

        return src_mask, tgt_mask

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)

            src_mask, tgt_mask = self.create_mask(src_ids, tgt_ids[:, :-1])
            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]

            output = self.model(src_ids, tgt_input, src_mask, tgt_mask)
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = self.criterion(output, tgt_output)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        batch_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_loader):
                src_ids = batch['src_ids'].to(self.device)
                tgt_ids = batch['tgt_ids'].to(self.device)

                src_mask, tgt_mask = self.create_mask(src_ids, tgt_ids[:, :-1])
                tgt_input = tgt_ids[:, :-1]
                tgt_output = tgt_ids[:, 1:]

                output = self.model(src_ids, tgt_input, src_mask, tgt_mask)
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1)
                loss = self.criterion(output, tgt_output)

                total_loss += loss.item()
                batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        self.valid_losses.append(avg_loss)
        return avg_loss

    def train(self, epochs, save_model=False):
        print("开始训练...")

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch()
            valid_loss = self.validate()

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch + 1}/{epochs} | Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
            print("-" * 50)

            # 保存模型
            if save_model:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'train_losses': self.train_losses,
                    'valid_losses': self.valid_losses,
                }, f'checkpoints/model_epoch_{epoch + 1}.pth')

        return self.train_losses, self.valid_losses


# ==================== 实验函数 ====================
def plot_training_curves(train_losses, valid_losses, save_path='outputs/training_curves.png'):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 5))

    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(valid_losses, label='Valid Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(model, test_loader, tokenizer, device, num_samples=10):
    """评估模型并计算简单的翻译质量分数"""
    model.eval()
    translations = []
    references = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break

            src_ids = batch['src_ids'].to(device)
            src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)
            reference = batch['tgt_text'][0]

            # 简单生成翻译（贪心搜索）
            max_len = 50
            start_token = 1
            end_token = 2

            generated = torch.tensor([[start_token]], device=device)

            for step in range(max_len):
                tgt_mask = model.generate_square_subsequent_mask(generated.size(1)).to(device)
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

                output = model(src_ids, generated, src_mask, tgt_mask)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == end_token:
                    break

            # 转换为文本
            translation_tokens = []
            for token_id in generated[0][1:]:
                if token_id.item() == end_token:
                    break
                if token_id.item() != 0:
                    token = tokenizer.id2token.get(token_id.item(), '<unk>')
                    if token not in ['<sos>', '<eos>', '<pad>', '<unk>']:
                        translation_tokens.append(token)

            candidate = ' '.join(translation_tokens)
            translations.append(candidate)
            references.append(reference.lower())

    # 计算简单的翻译质量分数（基于词汇重叠）
    scores = []
    for trans, ref in zip(translations, references):
        trans_tokens = set(trans.lower().split())
        ref_tokens = set(ref.split())
        if len(ref_tokens) > 0:
            overlap = len(trans_tokens & ref_tokens) / len(ref_tokens)
            scores.append(overlap)

    avg_score = np.mean(scores) if scores else 0.0

    return translations, references, avg_score


def hyperparameter_sensitivity_analysis(train_dataset, valid_dataset, src_vocab_size,
                                        tgt_vocab_size, device):
    """超参数敏感性分析"""
    print("\n" + "=" * 60)
    print("超参数敏感性分析")
    print("=" * 60)

    def collate_fn(batch):
        max_src_len = max(len(item['src_ids']) for item in batch)
        max_tgt_len = max(len(item['tgt_ids']) for item in batch)

        src_ids = torch.zeros(len(batch), max_src_len, dtype=torch.long)
        tgt_ids = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)

        for i, item in enumerate(batch):
            src_len = len(item['src_ids'])
            tgt_len = len(item['tgt_ids'])
            src_ids[i, :src_len] = item['src_ids']
            tgt_ids[i, :tgt_len] = item['tgt_ids']

        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids
        }

    results = {}

    # ==================== 1. 批量大小敏感性分析 ====================
    print("\n1. 批量大小敏感性分析")
    batch_sizes = [8, 16, 32]
    batch_results = []

    for batch_size in batch_sizes:
        print(f"\n  测试批量大小: {batch_size}")

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # 创建模型
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=256,
            dropout=0.1,
            max_len=30
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 训练
        trainer = Trainer(model, train_loader, valid_loader, optimizer, criterion, device)
        train_losses, valid_losses = trainer.train(epochs=2, save_model=False)

        final_loss = valid_losses[-1] if valid_losses else float('inf')

        batch_results.append({
            'batch_size': batch_size,
            'final_loss': final_loss,
            'train_time': len(train_losses)  # 简化的训练时间指标
        })

        print(f"    最终验证损失: {final_loss:.4f}")

    results['batch_size'] = batch_results

    # ==================== 2. 学习率敏感性分析 ====================
    print("\n2. 学习率敏感性分析")
    learning_rates = [0.00001, 0.0001, 0.001]
    lr_results = []

    # 固定批量大小
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    for lr in learning_rates:
        print(f"\n  测试学习率: {lr}")

        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=256,
            dropout=0.1,
            max_len=30
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        trainer = Trainer(model, train_loader, valid_loader, optimizer, criterion, device)
        train_losses, valid_losses = trainer.train(epochs=2, save_model=False)

        final_loss = valid_losses[-1] if valid_losses else float('inf')

        lr_results.append({
            'learning_rate': lr,
            'final_loss': final_loss
        })

        print(f"    最终验证损失: {final_loss:.4f}")

    results['learning_rate'] = lr_results

    # ==================== 3. 模型规模敏感性分析 ====================
    print("\n3. 模型规模敏感性分析")
    model_configs = [
        {'name': 'tiny', 'd_model': 64, 'n_layers': 1, 'n_heads': 2},
        {'name': 'small', 'd_model': 128, 'n_layers': 2, 'n_heads': 4},
        {'name': 'medium', 'd_model': 256, 'n_layers': 3, 'n_heads': 8},
    ]
    model_results = []

    for config in model_configs:
        print(f"\n  测试模型: {config['name']}")
        print(f"    配置: d_model={config['d_model']}, n_layers={config['n_layers']}, n_heads={config['n_heads']}")

        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_ff=config['d_model'] * 4,
            dropout=0.1,
            max_len=30
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        trainer = Trainer(model, train_loader, valid_loader, optimizer, criterion, device)
        train_losses, valid_losses = trainer.train(epochs=2, save_model=False)

        final_loss = valid_losses[-1] if valid_losses else float('inf')

        model_results.append({
            'name': config['name'],
            'd_model': config['d_model'],
            'n_layers': config['n_layers'],
            'n_heads': config['n_heads'],
            'params': sum(p.numel() for p in model.parameters()),
            'final_loss': final_loss
        })

        print(f"    参数量: {model_results[-1]['params']:,}")
        print(f"    最终验证损失: {final_loss:.4f}")

    results['model_size'] = model_results

    # ==================== 绘制敏感性分析图 ====================
    print("\n生成敏感性分析图表...")

    # 1. 批量大小敏感性
    fig, ax = plt.subplots(figsize=(8, 5))

    batch_sizes = [r['batch_size'] for r in results['batch_size']]
    batch_losses = [r['final_loss'] for r in results['batch_size']]

    ax.plot(batch_sizes, batch_losses, 'bo-', linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Batch Size Sensitivity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/sensitivity_batch_size.png', dpi=300)
    plt.close()

    # 2. 学习率敏感性
    fig, ax = plt.subplots(figsize=(8, 5))

    lr_values = [r['learning_rate'] for r in results['learning_rate']]
    lr_losses = [r['final_loss'] for r in results['learning_rate']]

    ax.plot(lr_values, lr_losses, 'go-', linewidth=2)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Learning Rate Sensitivity')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/sensitivity_learning_rate.png', dpi=300)
    plt.close()

    # 3. 模型规模敏感性
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    model_names = [r['name'] for r in results['model_size']]
    model_params = [r['params'] for r in results['model_size']]
    model_losses = [r['final_loss'] for r in results['model_size']]

    axes[0].bar(model_names, model_params, color=['blue', 'green', 'red'])
    axes[0].set_xlabel('Model Size')
    axes[0].set_ylabel('Number of Parameters')
    axes[0].set_title('Model Size vs Parameters')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].plot(model_names, model_losses, 'ro-', linewidth=2)
    axes[1].set_xlabel('Model Size')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Model Size vs Loss')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/sensitivity_model_size.png', dpi=300)
    plt.close()

    print("敏感性分析图表已保存到 outputs/ 目录")

    return results


# ==================== 预训练模型微调 ====================
def fine_tune_pretrained_model(train_dataset, valid_dataset, test_dataset, device):
    """使用预训练的T5模型进行微调"""
    print("\n" + "=" * 60)
    print("基于预训练模型的微调 (T5)")
    print("=" * 60)

    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        print("加载预训练的T5模型和分词器...")

        # 加载预训练模型
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

        # 准备少量数据用于演示
        print("准备微调数据...")

        train_samples = train_dataset.data[:100]  # 使用少量数据
        valid_samples = valid_dataset.data[:20]
        test_samples = test_dataset.data[:5]

        # 训练
        print("开始微调T5模型 (2个epoch)...")

        # 简化微调过程
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        train_losses = []
        for epoch in range(2):
            model.train()
            epoch_loss = 0

            for src_text, tgt_text in train_samples:
                # T5使用前缀进行翻译任务
                input_text = f"translate Chinese to English: {src_text}"
                target_text = tgt_text

                # 编码
                inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True).to(device)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(target_text, return_tensors="pt", max_length=64, truncation=True).to(device)

                # 前向传播
                outputs = model(**inputs, labels=labels['input_ids'])
                loss = outputs.loss

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_samples)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/2, Loss: {avg_loss:.4f}")

        # 测试
        print("\n测试微调后的T5模型...")
        model.eval()

        test_results = []
        with torch.no_grad():
            for src_text, tgt_text in test_samples:
                input_text = f"translate Chinese to English: {src_text}"

                inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True).to(device)

                outputs = model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )

                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

                test_results.append({
                    'source': src_text,
                    'reference': tgt_text,
                    'prediction': prediction
                })

                print(f"\n源文: {src_text}")
                print(f"参考: {tgt_text}")
                print(f"T5翻译: {prediction}")

        # 计算简单的翻译质量分数
        scores = []
        for result in test_results:
            pred_tokens = set(result['prediction'].lower().split())
            ref_tokens = set(result['reference'].lower().split())
            if len(ref_tokens) > 0:
                overlap = len(pred_tokens & ref_tokens) / len(ref_tokens)
                scores.append(overlap)

        avg_score = np.mean(scores) if scores else 0.0
        print(f"\nT5模型平均翻译质量分数: {avg_score:.4f}")

        t5_results = {
            'train_losses': train_losses,
            'test_results': test_results,
            'avg_score': avg_score
        }

        # 绘制训练曲线
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_losses, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('T5 Fine-tuning Training Loss')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/t5_training_loss.png', dpi=300)
        plt.close()

        print("T5训练曲线已保存到 outputs/t5_training_loss.png")

        return t5_results

    except ImportError:
        print("transformers库未安装，跳过T5微调实验。")
        print("如需运行此实验，请安装: pip install transformers")
        return None
    except Exception as e:
        print(f"T5微调出错: {e}")
        return None


# ==================== 主函数 ====================
def main():
    print("初始化...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("\n加载数据...")

    # 创建分词器
    src_tokenizer = SimpleTokenizer(lang='chinese')
    tgt_tokenizer = SimpleTokenizer(lang='english')

    # 读取训练数据构建词汇表
    train_src_texts = []
    train_tgt_texts = []

    with open('data/train_10k.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                keys = list(item.keys())
                if len(keys) >= 2:
                    train_src_texts.append(item[keys[0]])
                    train_tgt_texts.append(item[keys[1]])
            except:
                continue
            if i >= 999:
                break

    print(f"读取 {len(train_src_texts)} 条数据构建词汇表")

    # 构建词汇表
    src_tokenizer.build_vocab(train_src_texts, max_vocab_size=2000)
    tgt_tokenizer.build_vocab(train_tgt_texts, max_vocab_size=5000)

    print(f"中文词汇表大小: {src_tokenizer.vocab_size}")
    print(f"英文词汇表大小: {tgt_tokenizer.vocab_size}")

    # 创建数据集
    max_len = 30
    train_dataset = TranslationDataset('data/train_10k.jsonl', src_tokenizer, tgt_tokenizer, max_len=max_len)
    valid_dataset = TranslationDataset('data/valid.jsonl', src_tokenizer, tgt_tokenizer, max_len=max_len)
    test_dataset = TranslationDataset('data/test.jsonl', src_tokenizer, tgt_tokenizer, max_len=max_len)

    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(valid_dataset)} 条")
    print(f"测试集: {len(test_dataset)} 条")

    # ==================== 实验1: 基础Transformer训练 ====================
    print("\n" + "=" * 60)
    print("实验1: 从零训练Transformer模型")
    print("=" * 60)

    # 创建数据加载器
    def collate_fn(batch):
        max_src_len = max(len(item['src_ids']) for item in batch)
        max_tgt_len = max(len(item['tgt_ids']) for item in batch)

        src_ids = torch.zeros(len(batch), max_src_len, dtype=torch.long)
        tgt_ids = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)

        for i, item in enumerate(batch):
            src_len = len(item['src_ids'])
            tgt_len = len(item['tgt_ids'])
            src_ids[i, :src_len] = item['src_ids']
            tgt_ids[i, :tgt_len] = item['tgt_ids']

        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'src_text': [item['src_text'] for item in batch],
            'tgt_text': [item['tgt_text'] for item in batch]
        }

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 确保d_model能被n_heads整除
    d_model = 128
    n_heads = 4
    assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

    base_model = Transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=n_heads,
        d_ff=512,
        dropout=0.1,
        max_len=max_len
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in base_model.parameters()):,}")

    # 训练模型
    optimizer = optim.Adam(base_model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    trainer = Trainer(base_model, train_loader, valid_loader, optimizer, criterion, device)
    train_losses, valid_losses = trainer.train(epochs=3, save_model=True)

    # 绘制训练曲线
    os.makedirs('outputs', exist_ok=True)
    plot_training_curves(train_losses, valid_losses, 'outputs/transformer_training_curves.png')
    print("训练曲线已保存到 outputs/transformer_training_curves.png")

    # ==================== 实验2: 超参数敏感性分析 ====================
    hyperparameter_results = hyperparameter_sensitivity_analysis(
        train_dataset, valid_dataset, src_tokenizer.vocab_size,
        tgt_tokenizer.vocab_size, device
    )

    # ==================== 实验3: 基于预训练模型的微调 ====================
    t5_results = fine_tune_pretrained_model(train_dataset, valid_dataset, test_dataset, device)

    # ==================== 实验4: 性能对比 ====================
    print("\n" + "=" * 60)
    print("实验4: 模型性能对比")
    print("=" * 60)

    # 评估基础模型
    print("\n评估从零训练的Transformer模型...")
    base_model.eval()

    # 使用少量测试样本
    test_samples = []
    for i, batch in enumerate(test_loader):
        if i >= 5:
            break
        test_samples.append(batch)

    transformer_translations = []
    transformer_references = []

    with torch.no_grad():
        for i, batch in enumerate(test_samples):
            src_ids = batch['src_ids'].to(device)
            src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)
            reference = batch['tgt_text'][0]

            # 生成翻译
            max_len = 50
            start_token = 1
            end_token = 2

            generated = torch.tensor([[start_token]], device=device)

            for step in range(max_len):
                tgt_mask = base_model.generate_square_subsequent_mask(generated.size(1)).to(device)
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

                output = base_model(src_ids, generated, src_mask, tgt_mask)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == end_token:
                    break

            # 转换为文本
            translation_tokens = []
            for token_id in generated[0][1:]:
                if token_id.item() == end_token:
                    break
                if token_id.item() != 0:
                    token = tgt_tokenizer.id2token.get(token_id.item(), '<unk>')
                    if token not in ['<sos>', '<eos>', '<pad>', '<unk>']:
                        translation_tokens.append(token)

            candidate = ' '.join(translation_tokens)
            transformer_translations.append(candidate)
            transformer_references.append(reference)

            print(f"\n示例 {i + 1}:")
            print(f"  源文: {batch['src_text'][0]}")
            print(f"  参考翻译: {reference}")
            print(f"  Transformer翻译: {candidate}")

    # 计算基础模型的翻译质量分数
    transformer_scores = []
    for trans, ref in zip(transformer_translations, transformer_references):
        trans_tokens = set(trans.lower().split())
        ref_tokens = set(ref.lower().split())
        if len(ref_tokens) > 0:
            overlap = len(trans_tokens & ref_tokens) / len(ref_tokens)
            transformer_scores.append(overlap)

    avg_transformer_score = np.mean(transformer_scores) if transformer_scores else 0.0
    print(f"\n从零训练的Transformer模型平均翻译质量分数: {avg_transformer_score:.4f}")

    # 性能对比
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)

    print(f"\n1. 从零训练的Transformer模型:")
    print(f"   最终训练损失: {train_losses[-1]:.4f}")
    print(f"   最终验证损失: {valid_losses[-1]:.4f}")
    print(f"   测试集翻译质量分数: {avg_transformer_score:.4f}")

    if t5_results:
        print(f"\n2. 微调的T5模型:")
        print(f"   最终训练损失: {t5_results['train_losses'][-1]:.4f}")
        print(f"   测试集翻译质量分数: {t5_results['avg_score']:.4f}")

        print(f"\n3. 性能对比分析:")
        score_diff = t5_results['avg_score'] - avg_transformer_score
        print(f"   翻译质量分数差异: {score_diff:.4f}")
        if score_diff > 0:
            print(f"   T5模型表现更好，相对提升: {(score_diff / avg_transformer_score) * 100:.2f}%")
        else:
            print(f"   从零训练的Transformer模型表现更好")

    print(f"\n4. 超参数敏感性分析结果:")
    print(f"   最佳批量大小: {min(hyperparameter_results['batch_size'], key=lambda x: x['final_loss'])['batch_size']}")
    print(
        f"   最佳学习率: {min(hyperparameter_results['learning_rate'], key=lambda x: x['final_loss'])['learning_rate']}")
    best_model = min(hyperparameter_results['model_size'], key=lambda x: x['final_loss'])
    print(f"   最佳模型配置: {best_model['name']} (d_model={best_model['d_model']}, layers={best_model['n_layers']})")

    print(f"\n5. 可视化结果:")
    print("   - Transformer训练曲线: outputs/transformer_training_curves.png")
    print("   - 批量大小敏感性: outputs/sensitivity_batch_size.png")
    print("   - 学习率敏感性: outputs/sensitivity_learning_rate.png")
    print("   - 模型规模敏感性: outputs/sensitivity_model_size.png")
    if t5_results:
        print("   - T5训练曲线: outputs/t5_training_loss.png")

    print(f"\n所有实验完成!")


if __name__ == "__main__":
    main()