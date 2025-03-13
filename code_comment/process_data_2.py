import json
import torch
import torch.utils.data as Data
from collections import Counter
import re

# 分词函数（中文和英文）
def tokenize_zh(sentence):
    return list(sentence)  # 按字切分

def tokenize_en(sentence):
    return re.findall(r"\b\w+\b", sentence.lower())  # 以单词为单位切分

# 读取 JSON 数据并清理输入
def load_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                item = json.loads(line)
                item["input"] = item["input"].replace("请你把中文翻译成为英文\n", "")  # 移除前缀
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
    return data

# 构建词汇表
def build_vocab(sentences, tokenizer, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenizer(sentence))

    vocab = {word: idx + 4 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab.update({"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3})  # 预留特殊标记
    return vocab

# 将句子转换为数值化张量
def sentence_to_tensor(sentence, vocab, tokenizer, max_len):
    tokens = [vocab.get(word, vocab["<UNK>"]) for word in tokenizer(sentence)]
    tokens = [vocab["<SOS>"]] + tokens[: max_len - 2] + [vocab["<EOS>"]]
    tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

def make_data(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len):
    enc_inputs = [sentence_to_tensor(sent, src_vocab, tokenize_zh, max_len) for sent in src_sentences]
    dec_inputs = [sentence_to_tensor(sent, tgt_vocab, tokenize_en, max_len) for sent in tgt_sentences]

    # 🚀 `dec_outputs` 去掉 <SOS>，并保持长度一致
    dec_outputs = [torch.cat([tensor[1:], torch.tensor([tgt_vocab["<PAD>"]], dtype=torch.long)]) for tensor in dec_inputs]

    return torch.stack(enc_inputs), torch.stack(dec_inputs), torch.stack(dec_outputs)

# 自定义数据集
class MyDataset(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.size(0)

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

# collate_fn 处理不同长度的批次数据
def collate_fn(batch):
    enc_inputs, dec_inputs, dec_outputs = zip(*batch)
    return torch.stack(enc_inputs), torch.stack(dec_inputs), torch.stack(dec_outputs)


# 读取数据
file_path = 'train.json'
data = load_json(file_path)

# 处理句子
src_sentences = [item['input'].lstrip("请你把中文翻译成为英文\n") for item in data]  # 删除前缀
tgt_sentences = [item['output'] for item in data]

# 构建词汇表
src_vocab = build_vocab(src_sentences, tokenize_zh)
tgt_vocab = build_vocab(tgt_sentences, tokenize_en)

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# 计算最大长度
max_len = max(max(len(tokenize_zh(s)) for s in src_sentences), max(len(tokenize_en(t)) for t in tgt_sentences)) + 2
