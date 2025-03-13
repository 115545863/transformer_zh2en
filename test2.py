import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm
from process_data import *  # 确保包含必要的数据处理函数
from transformer import Transformer
from nltk.translate.bleu_score import sentence_bleu

# 加载测试数据
def load_test_data(file_path, src_vocab, tgt_vocab, max_len, max_samples=100):
    data = load_json_file(file_path)[:max_samples]  # 限制样本数
    src_sentences = [item['input'] for item in data]
    tgt_sentences = [item['output'] for item in data]
    enc_inputs, dec_inputs, dec_outputs = make_data(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    return Data.DataLoader(MyDataset(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=False, collate_fn=collate_fn)

# 评估模型
def evaluate_model(model, dataloader, criterion, tgt_vocab):
    model.eval()
    total_loss = 0
    total_bleu = 0
    num_samples = 0
    idx2word = {idx: word for word, idx in tgt_vocab.items()}

    with torch.no_grad():
        with tqdm(total=len(dataloader.dataset), desc="Evaluating", ncols=100) as pbar:
            for enc_inputs, dec_inputs, dec_outputs in dataloader:
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
                
                # 计算模型输出
                outputs, _, _, _ = model(enc_inputs, dec_inputs)  # (batch, seq_len, vocab_size)
                
                # 计算 loss
                loss = criterion(outputs.view(-1, outputs.size(-1)), dec_outputs.view(-1))
                total_loss += loss.item()
                
                # 计算 BLEU 分数
                for i in range(len(outputs)):
                    predicted_indices = outputs[i].argmax(dim=-1).tolist()  # 确保转换为列表
                    target_indices = dec_outputs[i].tolist()

                    # 转换索引到词
                    predicted_sentence = [idx2word[idx] for idx in predicted_indices if idx in idx2word]
                    target_sentence = [[idx2word[idx] for idx in target_indices if idx in idx2word]]

                    # 计算 BLEU 分数（避免空句子）
                    if predicted_sentence and target_sentence[0]:
                        bleu_score = sentence_bleu(target_sentence, predicted_sentence)
                        total_bleu += bleu_score

                    num_samples += 1

                pbar.update(len(enc_inputs))

    avg_loss = total_loss / len(dataloader)
    avg_bleu = total_bleu / num_samples if num_samples > 0 else 0
    return avg_loss, avg_bleu


if __name__ == "__main__":
    test_file_path = 'test.json'  # 测试数据集路径
    model_path = 'model\model_3.pth'  # 加载已训练模型
    
    # 加载词汇表（假设已经构建）
    # src_vocab = build_vocab_from_file('src_vocab.json')
    # tgt_vocab = build_vocab_from_file('tgt_vocab.json')
    max_len = 50  # 设定最大句子长度
    max_samples = 100  # 设定最大样本数
    
    # 加载测试数据
    test_loader = load_test_data(test_file_path, src_vocab, tgt_vocab, max_len, max_samples)
    
    # 加载模型
    model = torch.load(model_path).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 评估模型
    test_loss, test_bleu = evaluate_model(model, test_loader, criterion, tgt_vocab)
    print(f"Test Loss: {test_loss:.6f}, BLEU Score: {test_bleu:.4f}")