import json
import torch
import numpy as np
from torch.utils.data import DataLoader as Data
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tabulate import tabulate  # 用于生成规范的表格
import random
import string
from process_data_2 import *

# 读取数据集
def load_dataset(file_path, sample_size=50):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return random.sample(data, sample_size)

# 归一化文本（去除标点、转换小写）
def normalize_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

# 计算 BLEU 评分（增加平滑处理）
def compute_bleu(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    
    # 使用平滑函数，避免 BLEU 过低
    chencherry = SmoothingFunction()
    
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=chencherry.method1)

# 计算 ROUGE 评分
def compute_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]["rouge-l"]["f"]

# 解析预测结果
def decode_output(predictions, idx2word):
    predictions = np.atleast_2d(predictions)  # 确保至少是二维数据
    decoded_texts = []
    
    for sent in predictions:
        decoded_sent = []
        for idx in sent:
            word = idx2word.get(int(idx), '')  # 获取词汇
            if word in ["<EOS>", "</s>", "<PAD>"]:  # 遇到结束符或填充符停止
                break
            decoded_sent.append(word)
            
        decoded_texts.append(' '.join(decoded_sent))
    
    return decoded_texts


# 读取数据
file_path = 'test_small.json'
sample_data = load_dataset(file_path)

# 处理句子
src_sentences = [item['input'].lstrip("请你把中文翻译成为英文\n") for item in sample_data]  # 删除前缀
tgt_sentences = [item['output'] for item in sample_data]

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# 计算最大长度
max_len = max(max(len(tokenize_zh(s)) for s in src_sentences), max(len(tokenize_en(t)) for t in tgt_sentences)) + 2


# # 构建词汇表
# src_vocab = build_vocab(src_sentences, tokenizer=tokenize_zh)
# tgt_vocab = build_vocab(tgt_sentences, tokenizer=tokenize_en)
idx2word = {idx: word for word, idx in tgt_vocab.items()}

# 预处理数据
enc_inputs, dec_inputs, dec_outputs = make_data(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len)

# 发送到 GPU
enc_inputs = enc_inputs.cuda()
dec_inputs = dec_inputs.cuda()

# 加载模型
model = torch.load('model/model_translate_zh2en_80_c2.pth').cuda()
model.eval()

# 进行推理
with torch.no_grad():
    predictions, _, _, _ = model(enc_inputs, dec_inputs)
    predicted_indices = predictions.argmax(dim=-1).cpu().numpy()
    predicted_indices = predicted_indices.reshape(enc_inputs.shape[0], -1)  # 处理 batch 维度

    predicted_texts = decode_output(predicted_indices, idx2word)

# 计算 BLEU 和 ROUGE，并存储所有得分
bleu_scores = []
rouge_scores = []
table_data = []

for i in range(len(sample_data)):
    if i < len(predicted_texts):  # 确保索引不会超出范围
        ref_text = normalize_text(tgt_sentences[i])
        pred_text = normalize_text(predicted_texts[i])
        
        bleu = compute_bleu(ref_text, pred_text)
        rouge = compute_rouge(ref_text, pred_text)
        
        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        
        table_data.append([i+1, src_sentences[i], tgt_sentences[i], predicted_texts[i], round(bleu, 4), round(rouge, 4)])
    else:
        table_data.append([i+1, src_sentences[i], tgt_sentences[i], "无法获取翻译结果", 0.0, 0.0])

# 计算所有数据的平均 BLEU 和 ROUGE
avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

# 输出表格
headers = ["编号", "输入", "真实翻译", "模型翻译", "BLEU", "ROUGE-L"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# 记录日志
with open("evaluation_log.txt", "w", encoding="utf-8") as log_file:
    log_file.write(tabulate(table_data, headers=headers, tablefmt="grid"))
    log_file.write(f"\n总体 BLEU 平均分数: {round(avg_bleu, 4)}\n")
    log_file.write(f"总体 ROUGE-L 平均分数: {round(avg_rouge, 4)}\n")

# 输出总体评分
print(f"\n总体 BLEU 平均分数: {round(avg_bleu, 4)}")
print(f"总体 ROUGE-L 平均分数: {round(avg_rouge, 4)}")
