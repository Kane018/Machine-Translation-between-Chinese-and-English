import json
import os
import random
import time
from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# 1. 数据加载与预处理
# ----------------------------

class TranslationDataset(Dataset):
    def __init__(self, file_path: str, src_lang='en', tgt_lang='zh'):
        self.pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.pairs.append((data[src_lang], data[tgt_lang]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def tokenize(text: str, lang: str) -> List[str]:
    if lang == 'en':
        return nltk.word_tokenize(text.lower())
    elif lang == 'zh':
        return list(text.replace(' ', ''))  # 简单中文分字
    else:
        return text.split()


class Vocab:
    def __init__(self, min_freq=2):
        self.token2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.min_freq = min_freq

    def build(self, sentences: List[List[str]]):
        counter = Counter()
        for sent in sentences:
            counter.update(sent)
        idx = len(self.token2idx)
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2idx:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2idx.get(t, self.token2idx['<unk>']) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2token.get(i, '<unk>') for i in indices]


def collate_fn(batch, src_vocab, tgt_vocab):
    src_sents, tgt_sents = zip(*batch)
    src_tokens = [src_vocab.encode(tokenize(s, 'en')) for s in src_sents]
    tgt_tokens = [tgt_vocab.encode(tokenize(t, 'zh')) for t in tgt_sents]

    src_seq = [[src_vocab.token2idx['<sos>']] + s + [src_vocab.token2idx['<eos>']] for s in src_tokens]
    tgt_seq = [[tgt_vocab.token2idx['<sos>']] + t + [tgt_vocab.token2idx['<eos>']] for t in tgt_tokens]

    src_padded = pad_sequence([torch.tensor(s) for s in src_seq], batch_first=True,
                              padding_value=src_vocab.token2idx['<pad>'])
    tgt_padded = pad_sequence([torch.tensor(t) for t in tgt_seq], batch_first=True,
                              padding_value=tgt_vocab.token2idx['<pad>'])

    return src_padded.to(DEVICE), tgt_padded.to(DEVICE)


# ----------------------------
# 2. 注意力机制
# ----------------------------

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_type='additive'):
        super().__init__()
        self.attn_type = attn_type
        if attn_type == 'dot':
            pass
        elif attn_type == 'multiplicative':
            self.W = nn.Linear(enc_hidden_dim, dec_hidden_dim, bias=False)
        elif attn_type == 'additive':
            self.W = nn.Linear(enc_hidden_dim, dec_hidden_dim, bias=False)
            self.U = nn.Linear(dec_hidden_dim, dec_hidden_dim, bias=False)
            self.V = nn.Linear(dec_hidden_dim, 1, bias=False)
        else:
            raise ValueError("attn_type must be 'dot', 'multiplicative', or 'additive'")

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [src_len, batch, enc_hidden]
        # decoder_hidden: [batch, dec_hidden]
        src_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        dec_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, dec_hidden]
        enc_out = encoder_outputs.transpose(0, 1)  # [batch, src_len, enc_hidden]

        if self.attn_type == 'dot':
            energy = torch.bmm(dec_hidden, enc_out.transpose(1, 2))  # [batch, src_len, src_len] → 不对！修正：
            # 实际应为：[batch, 1, dec_hidden] × [batch, enc_hidden, src_len] → 需要维度匹配
            # 更正如下：
            energy = torch.sum(dec_hidden * enc_out, dim=2)  # [batch, src_len]
        elif self.attn_type == 'multiplicative':
            transformed_enc = self.W(enc_out)  # [batch, src_len, dec_hidden]
            energy = torch.sum(dec_hidden * transformed_enc, dim=2)  # [batch, src_len]
        elif self.attn_type == 'additive':
            energy = self.V(torch.tanh(self.W(enc_out) + self.U(dec_hidden))).squeeze(2)  # [batch, src_len]

        attn_weights = torch.softmax(energy, dim=1)  # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1)  # [batch, enc_hidden]
        return context, attn_weights


# ----------------------------
# 3. 编码器 & 解码器
# ----------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=2, dropout=0.3, rnn_type='GRU'):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        RNN = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        self.rnn = RNN(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch]
        embedded = self.dropout(self.embedding(src))  # [src_len, batch, emb_dim]
        outputs, hidden = self.rnn(embedded)  # outputs: [src_len, batch, hidden]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, n_layers=2,
                 dropout=0.3, attn_type='additive', rnn_type='GRU'):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.attention = Attention(enc_hidden_dim, dec_hidden_dim, attn_type)
        RNN = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        self.rnn = RNN(emb_dim + enc_hidden_dim, dec_hidden_dim, n_layers, dropout=dropout, batch_first=False)
        self.fc_out = nn.Linear(dec_hidden_dim + enc_hidden_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch]
        # hidden: [n_layers, batch, hidden] (for GRU) or tuple for LSTM
        input = input.unsqueeze(0)  # [1, batch]
        embedded = self.dropout(self.embedding(input))  # [1, batch, emb_dim]

        if isinstance(hidden, tuple):  # LSTM
            dec_hidden_for_attn = hidden[0][-1]  # [batch, hidden]
        else:  # GRU
            dec_hidden_for_attn = hidden[-1]  # [batch, hidden]

        context, attn_weights = self.attention(encoder_outputs, dec_hidden_for_attn)  # [batch, enc_hidden]
        context = context.unsqueeze(0)  # [1, batch, enc_hidden]

        rnn_input = torch.cat([embedded, context], dim=2)  # [1, batch, emb+enc]
        output, hidden = self.rnn(rnn_input, hidden)  # output: [1, batch, dec_hidden]
        output = output.squeeze(0)  # [batch, dec_hidden]
        context = context.squeeze(0)  # [batch, enc_hidden]
        embedded = embedded.squeeze(0)  # [batch, emb_dim]

        pred = self.fc_out(torch.cat([output, context, embedded], dim=1))  # [batch, output_dim]
        return pred, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        input = trg[0, :]  # <sos>
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


# ----------------------------
# 4. 训练 & 评估
# ----------------------------

def train(model, dataloader, optimizer, criterion, clip=1.0, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(dataloader, desc="Training"):
        src = src.transpose(0, 1)  # [src_len, batch]
        trg = trg.transpose(0, 1)  # [trg_len, batch]
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc="Evaluating"):
            src = src.transpose(0, 1)
            trg = trg.transpose(0, 1)
            output = model(src, trg, 0)  # no teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


# ----------------------------
# 5. 解码策略
# ----------------------------

def greedy_decode(model, src_tensor, src_vocab, tgt_vocab, max_len=50):
    model.eval()
    src_tensor = src_tensor.unsqueeze(1)  # [src_len, 1]
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        input_token = torch.tensor([tgt_vocab.token2idx['<sos>']]).to(DEVICE)
        decoded_tokens = []
        attentions = []
        for _ in range(max_len):
            output, hidden, attn = model.decoder(input_token, hidden, encoder_outputs)
            attentions.append(attn.squeeze(0).cpu().numpy())
            pred_token = output.argmax(1)
            if pred_token.item() == tgt_vocab.token2idx['<eos>']:
                break
            decoded_tokens.append(pred_token.item())
            input_token = pred_token
        return decoded_tokens, np.array(attentions)


def beam_search_decode(model, src_tensor, src_vocab, tgt_vocab, beam_width=3, max_len=50):
    model.eval()
    src_tensor = src_tensor.unsqueeze(1)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        hypotheses = [{'tokens': [tgt_vocab.token2idx['<sos>']], 'score': 0.0, 'hidden': hidden}]
        completed = []

        for step in range(max_len):
            new_hyps = []
            for hyp in hypotheses:
                if hyp['tokens'][-1] == tgt_vocab.token2idx['<eos>']:
                    completed.append(hyp)
                    continue
                input_token = torch.tensor([hyp['tokens'][-1]]).to(DEVICE)
                output, new_hidden, _ = model.decoder(input_token, hyp['hidden'], encoder_outputs)
                log_probs = torch.log_softmax(output, dim=1).squeeze(0)
                topk_vals, topk_idxs = log_probs.topk(beam_width)
                for val, idx in zip(topk_vals, topk_idxs):
                    new_hyps.append({
                        'tokens': hyp['tokens'] + [idx.item()],
                        'score': hyp['score'] + val.item(),
                        'hidden': new_hidden
                    })
            hypotheses = sorted(new_hyps, key=lambda x: x['score'], reverse=True)[:beam_width]
        if not completed:
            completed = hypotheses
        best = max(completed, key=lambda x: x['score'] / len(x['tokens']))
        return best['tokens'][1:], None  # remove <sos>


# ----------------------------
# 6. 可视化 & 结果分析
# ----------------------------

def plot_attention(src_sent, pred_tokens, attention, src_vocab, tgt_vocab, filename):
    src_words = ['<sos>'] + src_vocab.decode(src_sent.tolist()) + ['<eos>']
    tgt_words = ['<sos>'] + tgt_vocab.decode(pred_tokens) + ['<eos>']
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention[:len(tgt_words), :len(src_words)], annot=True, fmt='.2f',
                xticklabels=src_words, yticklabels=tgt_words, ax=ax)
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def calculate_bleu(pred_tokens, ref_tokens):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)


# ----------------------------
# 7. 主流程
# ----------------------------

def main():
    # 加载数据
    train_dataset = TranslationDataset('data/train_10k.jsonl')
    valid_dataset = TranslationDataset('data/valid.jsonl')
    test_dataset = TranslationDataset('data/test.jsonl')

    # 构建词表
    all_src = [tokenize(pair[0], 'en') for pair in train_dataset.pairs]
    all_tgt = [tokenize(pair[1], 'zh') for pair in train_dataset.pairs]
    src_vocab = Vocab(min_freq=2)
    tgt_vocab = Vocab(min_freq=2)
    src_vocab.build(all_src)
    tgt_vocab.build(all_tgt)

    print(f"Source vocab size: {len(src_vocab.token2idx)}")
    print(f"Target vocab size: {len(tgt_vocab.token2idx)}")

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab))
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab))

    # 超参数
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    RNN_TYPE = 'GRU'

    # 对比不同注意力机制
    attn_types = ['additive', 'dot', 'multiplicative']
    results = {}

    for attn_type in attn_types:
        print(f"\n=== Training with {attn_type} attention ===")
        encoder = Encoder(len(src_vocab.token2idx), ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, RNN_TYPE)
        decoder = Decoder(len(tgt_vocab.token2idx), DEC_EMB_DIM, HID_DIM, HID_DIM, N_LAYERS,
                          DEC_DROPOUT, attn_type, RNN_TYPE)
        model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.token2idx['<pad>'])

        train_losses, valid_losses = [], []
        best_valid_loss = float('inf')
        for epoch in range(10):
            train_loss = train(model, train_loader, optimizer, criterion, teacher_forcing_ratio=0.5)
            valid_loss = evaluate(model, valid_loader, criterion)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.3f}, Val Loss: {valid_loss:.3f}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{attn_type}.pt'))

        # Plot training curve
        plt.figure()
        plt.plot(train_losses, label='Train')
        plt.plot(valid_losses, label='Valid')
        plt.title(f'Training Curve ({attn_type})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f'train_curve_{attn_type}.png'))
        plt.close()

        # 测试集 BLEU 评估（贪心 vs beam）
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'model_{attn_type}.pt')))
        bleu_greedy, bleu_beam = [], []
        for i, (src, trg) in enumerate(test_loader):
            if i >= 10: break  # 仅测试前10句用于可视化
            src = src.squeeze(0)
            trg = trg.squeeze(0)
            pred_g, attn = greedy_decode(model, src, src_vocab, tgt_vocab)
            pred_b, _ = beam_search_decode(model, src, src_vocab, tgt_vocab, beam_width=3)
            ref = [idx for idx in trg.tolist() if idx not in [0, 1, 2]]
            bleu_greedy.append(calculate_bleu(pred_g, ref))
            bleu_beam.append(calculate_bleu(pred_b, ref))
            if i == 0 and attn is not None:
                plot_attention(src, pred_g, attn, src_vocab, tgt_vocab,
                               os.path.join(OUTPUT_DIR, f'attention_{attn_type}.png'))

        results[attn_type] = {
            'greedy_bleu': np.mean(bleu_greedy),
            'beam_bleu': np.mean(bleu_beam)
        }

    # 输出结果
    print("\n=== Final Results ===")
    for attn, res in results.items():
        print(f"{attn}: Greedy BLEU={res['greedy_bleu']:.4f}, Beam BLEU={res['beam_bleu']:.4f}")

    # Teacher Forcing vs Free Running 对比（以 additive 为例）
    print("\n=== Teacher Forcing vs Free Running (additive) ===")
    encoder = Encoder(len(src_vocab.token2idx), ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, RNN_TYPE)
    decoder = Decoder(len(tgt_vocab.token2idx), DEC_EMB_DIM, HID_DIM, HID_DIM, N_LAYERS,
                      DEC_DROPOUT, 'additive', RNN_TYPE)
    model_tf = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    optimizer = optim.Adam(model_tf.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.token2idx['<pad>'])

    losses_tf, losses_fr = [], []
    for epoch in range(5):
        loss_tf = train(model_tf, train_loader, optimizer, criterion, teacher_forcing_ratio=1.0)
        loss_fr = train(model_tf, train_loader, optimizer, criterion, teacher_forcing_ratio=0.0)
        losses_tf.append(loss_tf)
        losses_fr.append(loss_fr)
    plt.figure()
    plt.plot(losses_tf, label='Teacher Forcing')
    plt.plot(losses_fr, label='Free Running')
    plt.title('Teacher Forcing vs Free Running')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tf_vs_fr.png'))
    plt.close()


if __name__ == '__main__':
    main()