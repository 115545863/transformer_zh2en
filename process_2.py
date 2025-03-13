import json
import torch
import torch.utils.data as Data
from collections import Counter
import jieba

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
                print(f"Error: {e}")
    return data

# 分词函数
def tokenize_zh(text):
    return list(jieba.cut(text))

def tokenize_en(text):
    return text.split()

# 构建词汇表
def build_vocab(sentences, tokenizer):
    counter = Counter()
    for sentence in sentences:
        tokens = tokenizer(sentence)
        counter.update(tokens)
    vocab = {'<pad>': 0, '<s>': 1, '</s>': 2}  # 添加起始符号和结束符号
    for word, freq in counter.items():
        if freq > 1:
            vocab[word] = len(vocab)
    return vocab

# 转换为索引
def make_data(src_sentences, tgt_sentences, src_vocab, tgt_vocab):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    
    for src, tgt in zip(src_sentences, tgt_sentences):
        src_tokens = tokenize_zh(src)
        tgt_tokens = tokenize_en(tgt)

        # 编码器输入
        enc_input = [src_vocab.get(word, src_vocab['<pad>']) for word in src_tokens]

        # 解码器输入
        dec_input = [tgt_vocab['<s>']] + [tgt_vocab.get(word, tgt_vocab['<pad>']) for word in tgt_tokens]

        # 解码器输出（句尾添加 </s>）
        dec_output = [tgt_vocab.get(word, tgt_vocab['<pad>']) for word in tgt_tokens] + [tgt_vocab['</s>']]

        enc_inputs.append(torch.tensor(enc_input, dtype=torch.long))
        dec_inputs.append(torch.tensor(dec_input, dtype=torch.long))
        dec_outputs.append(torch.tensor(dec_output, dtype=torch.long))

    return enc_inputs, dec_inputs, dec_outputs

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

# collate_fn，保证填充一致
def collate_fn(batch):
    enc_inputs, dec_inputs, dec_outputs = zip(*batch)
    enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=src_vocab['<pad>'])
    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=tgt_vocab['<pad>'])
    dec_outputs = torch.nn.utils.rnn.pad_sequence(dec_outputs, batch_first=True, padding_value=tgt_vocab['<pad>'])
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

# 生成 idx2word 映射
idx2word = {idx: word for word, idx in tgt_vocab.items()}

# 设置最大长度
max_len = max(len(tokenize_zh(src)) for src in src_sentences) + 1  # 句子最大长度，加上开始符号
max_len = max(max_len, max(len(tokenize_en(tgt)) for tgt in tgt_sentences) + 1)  # 目标句子最大长度，加上开始符号

# 生成数据
enc_inputs, dec_inputs, dec_outputs = make_data(src_sentences, tgt_sentences, src_vocab, tgt_vocab)

# 构建数据集和 DataLoader
dataset = MyDataset(enc_inputs, dec_inputs, dec_outputs)
dataloader = Data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

print("Source Vocabulary Size:", len(src_vocab))
print("Target Vocabulary Size:", len(tgt_vocab))
print("Max Sentence Length:", max_len)

# 调试：打印部分数据
print("示例数据:")
print("enc_inputs[0]:", enc_inputs[0])
print("dec_inputs[0]:", dec_inputs[0])
print("dec_outputs[0]:", dec_outputs[0])
