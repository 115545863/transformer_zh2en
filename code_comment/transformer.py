import numpy as np
import torch.nn as nn
from process_data_2 import *

d_model = 256   # 字 Embedding 的维度
d_ff = 1024     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 4    # 有多少个encoder和decoder
n_heads = 4     # Multi-Head Attention设置为8


class PositionalEncoding(nn.Module):
    # 位置编码
    # 这个其实是个值，为什么要用这个值呢，因为transformer是并行计算的，位置信息比较微弱，所以需要一个位置编码来引入位置信息
    def __init__(self, d_model, dropout=0.3, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 位置编码公式
        # 对偶数: PE_{pos,i} = sin(pos/10000^{2*i/d_{model}})
        # 对奇数：PE_{pos,i} = cos(pos/10000^{2*i/d_{model}})
        # np.array将列表转化为numpy数组
        # 二维数组，行是某个字的多个维度（这里为8；列为最大长度（这个地方要注意，因为并不是每次都能达到最大长度的，所以其实不是每次都要全部用
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])           # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])           # 字嵌入维度为奇数时
        # 将pos_table转化为pytorch张量，然后将张量移动到gpu（.cuda方法），将结果储存在类属性pos_table上
        self.pos_table = torch.FloatTensor(pos_table).cuda()        # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):                                  # enc_inputs: [batch_size, seq_len, d_model]
        # forward方法是用来传递数据给下一层的
        # enc_inputs.size(1)返回序列长度seq_len
        # 为什么要限制长度，因为pos_table初始化的时候使用的是max_len，不是当前序列的长度，所以每次使用的时候要使用序列的长度，也就是seq_len
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        # dropout是用来随机丢弃一部分神经元的，防止过拟合
        # enc_inputs.cuda()将数据移动到GPU上
        return self.dropout(enc_inputs.cuda())


def get_attn_pad_mask(seq_q, seq_k):                                # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    # 填充掩码
    # 因为在实际处理问题的过程中，输入的序列的大小不是固定的，换句话说，一句话有的时候是5个字，有的时候是10个字
    # 所以我们其实在处理的时候使用最长序列进行执行，但是因为序列不是每次都能达到最长序列，因此我们要把不够最长序列的其他地方填充上值，这个值还不能影响后续进行操作的时候
    # 所以其实是填充上一个很小的值防止影响到
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # seq_k.data获取张量的底层数据
    # eq(0)，用来检测填充的位置，因为一般来说用0来填充的，这个会生成一个布尔组
    # 如果该位置是0，不参与计算，说明这个位置是被填充的，也就是返回True，否则返回False
    # unsqueeze(x)，在指定位置x增加一个维度
    # unsqueeze(1)，也就是在index=1的位置增加一个大小为1的维度
    # 原来是[betch_size,len_k]，现在是[betch_size,1,len_k]
    # 用于在注意力分数屏蔽填充的位置，防止对填充位置进行计算，此外，还要补上一个维度，是为了满足计算的矩阵格式需求，所以要补上一个
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)                   # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    # 1. 为什么不直接扩展，因为方法不允许，会报错，张量形状不一致的话不能直接扩展
    # 2. 先增加再扩展就满足要求了，然后注意力分数计算的形状通常是[batch_size, len_q, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)           # 扩展成多维度


def get_attn_subsequence_mask(seq):                                 # seq: [batch_size, tgt_len]
    # tgt是目标序列的长度
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # np.ones(attn_shape)是设置一个形状为attn_shape的矩阵的值为1
    # np.triu(..., k=1)，设置上三角，k=1是指对于对角线以上的部分保留为1，其余部分设置为0
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)            # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    # 将numpy转化为pytorch张量
    # .byte()能够将张量的数据类型转化为torch.uint8，更适合作为掩码
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()    # [batch_size, tgt_len, tgt_len]
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):                              # Q: [batch_size, n_heads, len_q, d_q]
                                                                        # K: [batch_size, n_heads, len_k, d_k]
                                                                        # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        # torch.matmul矩阵惩罚
        #  K.transpose(-1, -2)将k的后两个维度交换位置
        # 比如[batch_size, n_heads, len_k, d_k]，转化之后[batch_size, n_heads, d_k, len_k]
        # 为什么交换位置？因为做乘法的时候是k和q进行相乘，所以要让len_q乘d_k,len_k乘d_q
        # 乘完之后，变成了[batch_size, n_heads, len_q, len_k]
        # 除以 np.sqrt(d_k) 的目的是缩放点积（Scaled Dot-Product），以保持梯度的稳定
        # 如果d_k太大的话，会影响结果，所以要把结果缩放到一个合理的范围，使用缩放点积
        # 为什么使用根号dk?主要是考虑到了方差，点乘的方差受到dk的影响，除了根号dk，点乘的效果变化不会太大，保持数值稳定
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)    # scores : [batch_size, n_heads, len_q, len_k]

        # 如果是true，就用-1e9填充，因为这样的话数很小，不会影响到结果
        # masked_fill_ 是 PyTorch 的一个方法，用于将张量中被掩码的位置设置为指定的值。
        # 它会原地修改（in-place）张量，因此方法名后有一个下划线
        scores.masked_fill_(attn_mask, -1e9)                            # 如果时停用词P就等于 0

        # 为什么只处理最后一个维度，是因为我们这个地方需要的是查询对于每一个key的相关性权重
        # 换句话说，可以理解为考虑维度x，也就是其他值相对于x的相关性
        # 这里的意思是查询相关于所有k的相关性，因为选择的维度是-1，而-1刚好是键的关系
        # 如果是-2，刚好是q的位置，也就是其他值相对于所有q的相关性分数，但是这里没有意义，因为这里用于判断查询的相关性的
        attn = nn.Softmax(dim=-1)(scores)
        # 相关性分数和内容做矩阵乘法
        context = torch.matmul(attn, V)                                 # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # nn.linear是线性变换层，实现全连接层的功能，他的公式是y=XW^T+b,其中，X是输入，W是权重矩阵，b是偏置向量
        # d_model是模型的输入维度；d_k*n_heads是向量的输出维度，表示所有头的查询向量的总维度
        # 为什么输入一个维度，输出的时候会改变为另一个维度呢，因为其实在做线性变换的时候，输出的维度作为权重矩阵的行数，所以乘完之后会按照输出要求的维度进行输出
        # linear只处理最后一个维度
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # 多头注意力的输出重新映射回模型的原始维度
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):    # input_Q: [batch_size, len_q, d_model]
                                                                # input_K: [batch_size, len_k, d_model]
                                                                # input_V: [batch_size, len_v(=len_k), d_model]
                                                                # attn_mask: [batch_size, seq_len, seq_len]
        # residual 残差
        residual, batch_size = input_Q, input_Q.size(0)
        # 只处理最后一个维度，所以也就是输入的是input_Q但是也只是处理input_Q的最后一个维度d_model，确实也符合要求
        # view(batch_size, -1, n_heads, d_k),-1的意思是说，我不确定这个地方的具体值，我使用其他值，然后让算法自动计算-1这个位置的值，然后自动处理
        # view是用来重塑张量
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)    # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)    # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)       # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # 在index=1的位置插入一个值为1的维度
        # repeat的意思是，位置为1的不变（重复1次＝不变）,其他值就是重复多少次，下面是重复n_head次，也就是刚刚的那个插入的1重复n_head次
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,1)                                # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # 计算注意力分数
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)             # context: [batch_size, n_heads, len_q, d_v]
                                                                                    # attn: [batch_size, n_heads, len_q, len_k]
        # 交换位置，并且重构结构，因为contex输出的结构为[batch_size, n_heads, len_q, d_v]，这个是多个头的，我要重新把所有头拼接
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)                    # context: [batch_size, len_q, n_heads * d_v]
        # 使用线性变换层整合，把多头重新映射会模型的原始维度
        output = self.fc(context)                                                   # [batch_size, len_q, d_model]

        # output是输出，residual是输入
        # 梯度是一个向量，表示函数在某一点的变化率（即导数）和变化方向。在多维空间中，梯度是一个向量，指向函数值增长最快的方向。
        # 残差连接（输入＋输出）为梯度提供了一个直接的传播路径，使得梯度可以直接从输出传播到输入，而不需要经过复杂的网络层。
        # nn.LayerNorm创建一个层归一化模块，对每个输入向量的最后一个维度（d_model）进行归一化。
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    # 逐位置前馈网络
    # 在每个位置上对输入数据进行非线性变换。它的设计目标是在每个序列位置上独立地应用前馈网络，从而增加模型的非线性能力和表达能力
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # 是 PyTorch 中的一个容器，用于将多个模块（如线性层、激活函数等）按顺序组合成一个模块。
        #
        self.fc = nn.Sequential(
            # 线性层
            nn.Linear(d_model, d_ff, bias=False),
            # 激活函数
            # 用于引入非线性
            # ReLU 的优点是计算简单，且能够有效缓解梯度消失问题。
            # GeLU 是一种平滑的非线性激活函数，能够更好地捕捉输入的非线性关系。
            # 不同的激活函数效果不同
            # 因为不引入非线性的话，好多复杂的关系不容易表现出来
            nn.GELU(),  # 默认激活函数
            # 线性层
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):                                  # inputs: [batch_size, seq_len, d_model]
        # 输入
        residual = inputs
        # 输出
        output = self.fc(inputs)
        # 归一化
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                   # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                      # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):              # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V            # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                                    # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)  # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)                     # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()       # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()          # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):  # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V             # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                                        # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)      # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)                         # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)                     # 把字转换字向量
        self.pos_emb = PositionalEncoding(d_model)                               # 加入位置信息
        #  ModuleList([EncoderLayer() for _ in range(n_layers)]) 可以用于存储多个编码器或解码器层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):                                               # enc_inputs: [batch_size, src_len]
        # 对输入进行编码
        enc_outputs = self.src_emb(enc_inputs)                                   # enc_outputs: [batch_size, src_len, d_model]
        # 引入位置信息
        enc_outputs = self.pos_emb(enc_outputs)                                  # enc_outputs: [batch_size, src_len, d_model]
        # 获得掩码位置
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)           # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        # 存储多个结果
        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                dec_enc_attn_mask):                                             # dec_inputs: [batch_size, tgt_len, d_model]
                                                                                # enc_outputs: [batch_size, src_len, d_model]
                                                                                # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
                                                                                # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs,
                                                        dec_self_attn_mask)     # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs,
                                                      dec_enc_attn_mask)        # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)                                 # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):                         # dec_inputs: [batch_size, tgt_len]
                                                                                    # enc_intpus: [batch_size, src_len]
                                                                                    # enc_outputs: [batsh_size, src_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs)                                      # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs).cuda()                              # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()   # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        # torch.gt 是 PyTorch 中的一个函数，用于逐元素比较张量。
        # torch.gt(tensor, 0) 返回一个布尔张量，其中大于 0 的位置为 True，否则为 False
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0).cuda()   # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)               # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:                                                   # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                    # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
                                                                                    # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder().cuda()
        self.Decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs):                          # enc_inputs: [batch_size, src_len]
                                                                        # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)          # enc_outputs: [batch_size, src_len, d_model],
                                                                        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)                        # dec_outpus    : [batch_size, tgt_len, d_model],
                                                                        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
                                                                        # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        # 将解码输出值进行线性变换
        dec_logits = self.projection(dec_outputs)                       # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # 将结果重构为长度是目标值的长度，
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
