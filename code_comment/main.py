import json
import torch
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm  # 用于动态进度条
import torch.nn as nn
import os
from process_data_2 import *  # 假设已经包含了必要的数据处理函数
from transformer import Transformer

# 创建日志文件夹并定义日志文件路径
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 自动编号日志文件
log_files = [f for f in os.listdir(log_dir) if f.startswith('train_log')]
log_file_num = len(log_files) + 1
log_file_path = os.path.join(log_dir, f'train_log_{log_file_num}.txt')

# 定义日志写入函数
def write_log(log_message):
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(log_message + '\n')

# 自动编号模型文件
model_files = [f for f in os.listdir() if f.startswith('model')]
model_file_num = len(model_files) + 1
model_file_path = f'model/model_translate_50_{model_file_num}.pth'

if __name__ == "__main__":



    # 生成数据
    enc_inputs, dec_inputs, dec_outputs, src_vocab_size, tgt_vocab_size = make_data(
    src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len
)

    # 构建 DataLoader
    dataset = MyDataset(enc_inputs, dec_inputs, dec_outputs)
    loader = Data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


    # 初始化模型、损失函数、优化器
    model = Transformer().cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略占位符索引为 0
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)



    total_batches = len(loader)  # 获取总批次数
    total_samples = len(enc_inputs)  # 获取数据总数（样本数）

    for epoch in range(80):
        total_loss = 0  # 用于计算平均 loss
        num_batches = len(loader)  # 获取批次数量

        # 使用 tqdm 显示进度条，设置总进度为 total_samples
        with tqdm(total=total_samples, desc=f"Epoch {epoch + 1}", ncols=100) as pbar:
            for batch_idx, (enc_inputs, dec_inputs, dec_outputs) in enumerate(loader):
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()

                # 前向传播
                outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

                # 计算 loss
                loss = criterion(outputs, dec_outputs.view(-1))
                total_loss += loss.item()  # 累加当前 batch 的 loss

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新进度条描述
                avg_loss = total_loss / (batch_idx + 1)  # 计算平均 loss
                pbar.set_postfix(loss=f'{avg_loss:.6f}')

                # 每个 batch 完成后更新进度条
                pbar.update(len(enc_inputs))  # 更新进度条的进度，每个批次的大小
        scheduler.step()

        # 每个 epoch 结束后，保存 loss 信息到 log 文件
        epoch_avg_loss = total_loss / num_batches
        log_message = f"Epoch {epoch + 1}, Average Loss: {epoch_avg_loss:.6f}"
        write_log(log_message)

        # 打印每个 epoch 的平均 loss
        print(f'Epoch {epoch + 1}, Average Loss: {epoch_avg_loss:.6f}')

    # 保存模型
    torch.save(model, model_file_path)
    print(f"模型已保存到 {model_file_path}")
