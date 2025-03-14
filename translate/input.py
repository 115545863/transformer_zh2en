# import json
# import torch
# import numpy as np
# from process_data_2 import *  # 假设这个模块包含了必要的预处理函数

# # 归一化文本（去除标点、转换小写）
# def normalize_text(text):
#     import string
#     return text.lower().translate(str.maketrans('', '', string.punctuation))

# # 解析预测结果
# def decode_output(predictions, idx2word):
#     predictions = np.atleast_2d(predictions)  # 确保至少是二维数据
#     decoded_texts = []
    
#     for sent in predictions:
#         decoded_sent = []
#         for idx in sent:
#             word = idx2word.get(int(idx), '')  # 获取词汇
#             if word in ["<EOS>", "</s>", "<PAD>"]:  # 遇到结束符或填充符停止
#                 break
#             decoded_sent.append(word)
            
#         decoded_texts.append(' '.join(decoded_sent))
    
#     return decoded_texts

# # 加载词汇表
# # 假设 tgt_vocab 是目标语言的词汇表，idx2word 是索引到单词的映射
# # tgt_vocab = json.load(open('tgt_vocab.json', 'r', encoding='utf-8'))  # 加载目标语言词汇表
# idx2word = {idx: word for word, idx in tgt_vocab.items()}

# # 加载模型
# model = torch.load('model/model_translate_zh2en_80_c2.pth').cuda()
# model.eval()

# # 用户输入
# user_input = input("请输入中文句子：")

# # 对用户输入进行预处理
# src_sentence = user_input.lstrip("请你把中文翻译成为英文\n")  # 删除前缀（如果有的话）
# src_tokens = tokenize_zh(src_sentence)  # 对用户输入进行分词
# max_len = len(src_tokens) + 2  # 计算最大长度，+2 是为了留出空间给特殊标记（如 <SOS> 和 <EOS>）

# # 构建词汇表（假设 build_vocab 和 make_data 是必要的预处理函数）
# src_vocab = build_vocab([src_sentence], tokenizer=tokenize_zh)  # 构建用户输入的词汇表

# # 预处理数据
# enc_inputs, dec_inputs, _ = make_data([src_sentence], [""], src_vocab, tgt_vocab, max_len)

# # 发送到 GPU
# enc_inputs = enc_inputs.cuda()
# dec_inputs = dec_inputs.cuda()

# # 进行推理
# with torch.no_grad():
#     predictions, _, _, _ = model(enc_inputs, dec_inputs)
#     predicted_indices = predictions.argmax(dim=-1).cpu().numpy()
#     predicted_indices = predicted_indices.reshape(enc_inputs.shape[0], -1)  # 处理 batch 维度

#     predicted_texts = decode_output(predicted_indices, idx2word)

# # 输出结果
# print(f"用户输入：{user_input}")
# print(f"模型翻译：{predicted_texts[0]}")