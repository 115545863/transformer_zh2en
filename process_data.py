import json
import torch
import torch.utils.data as Data
from collections import Counter
import jieba
import re
import string
import os

# 读取 JSON 文件
def load_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}")
    return data

# 中文分词（使用 jieba）
def tokenize_zh(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除所有标点
    return list(jieba.cut(text))  # 用 jieba 进行分词

# 英文分词（简单按空格拆分）
def tokenize_en(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))  # 转小写 & 去标点
    return text.split()  # 按空格拆分

# 构建词汇表
def build_vocab(sentences, tokenizer, vocab_size=50000, min_freq=2):
    counter = Counter()
    for sentence in sentences:
        tokens = tokenizer(sentence)
        counter.update(tokens)
    
    # 限制词汇表大小，低频单词映射到 <UNK>
    most_common = counter.most_common(vocab_size - 4)  # 预留 <pad>, <s>, </s>, <unk>
    vocab = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3}
    
    for word, freq in most_common:
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab

# 转换为索引
def make_data(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=50):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    
    for src, tgt in zip(src_sentences, tgt_sentences):
        src_tokens = tokenize_zh(src)
        tgt_tokens = tokenize_en(tgt)

        # 编码器输入
        enc_input = [src_vocab.get(word, src_vocab['<unk>']) for word in src_tokens]
        enc_input = [src_vocab['<s>']] + enc_input[:max_len-1]  # 添加开始符号，限制最大长度
        enc_input += [src_vocab['<pad>']] * (max_len - len(enc_input))

        # 解码器输入
        dec_input = [tgt_vocab['<s>']] + [tgt_vocab.get(word, tgt_vocab['<unk>']) for word in tgt_tokens][:max_len-1]
        dec_input += [tgt_vocab['<pad>']] * (max_len - len(dec_input))

        # 解码器输出
        dec_output = [tgt_vocab.get(word, tgt_vocab['<unk>']) for word in tgt_tokens][:max_len-1] + [tgt_vocab['</s>']]
        dec_output += [tgt_vocab['<pad>']] * (max_len - len(dec_output))

        enc_inputs.append(torch.tensor(enc_input, dtype=torch.long))
        dec_inputs.append(torch.tensor(dec_input, dtype=torch.long))
        dec_outputs.append(torch.tensor(dec_output, dtype=torch.long))

    return torch.stack(enc_inputs), torch.stack(dec_inputs), torch.stack(dec_outputs)

# 数据集类
class MyDataset(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataset, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return len(self.enc_inputs)

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

# Collate 函数，保证填充一致
def collate_fn(batch):
    enc_inputs, dec_inputs, dec_outputs = zip(*batch)
    enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=0)
    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)
    dec_outputs = torch.nn.utils.rnn.pad_sequence(dec_outputs, batch_first=True, padding_value=0)
    return enc_inputs, dec_inputs, dec_outputs

# 读取数据
file_path = 'train.json'
data = load_json_file(file_path)

# 处理句子
src_sentences = [item['input'].lstrip("请你把中文翻译成为英文\n") for item in data]  # 删除前缀
tgt_sentences = [item['output'] for item in data]

# 构建词汇表
src_vocab = build_vocab(src_sentences, tokenizer=tokenize_zh)
tgt_vocab = build_vocab(tgt_sentences, tokenizer=tokenize_en)

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

print(f"✅ 源语言词汇表大小: {len(src_vocab)}")
print(f"✅ 目标语言词汇表大小: {len(tgt_vocab)}")

# 生成 idx2word 映射
idx2word = {idx: word for word, idx in tgt_vocab.items()}

# 设置最大长度
max_len = 50  # 固定最大长度，避免过长

# 生成数据
enc_inputs, dec_inputs, dec_outputs = make_data(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len)

# 构建数据集和 DataLoader
dataset = MyDataset(enc_inputs, dec_inputs, dec_outputs)
dataloader = Data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

print("📊 训练数据集构建完成！")
print(f"📏 句子最大长度: {max_len}")
print(f"📝 示例数据:\n  🔹 enc_inputs[0]: {enc_inputs[0]}\n  🔹 dec_inputs[0]: {dec_inputs[0]}\n  🔹 dec_outputs[0]: {dec_outputs[0]}")
