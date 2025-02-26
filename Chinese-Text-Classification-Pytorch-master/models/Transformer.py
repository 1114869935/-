import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'Transformer'  # 模型名字
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 搞一个类别名单
        # 词表的作用是将中文转化为一个个向量，从而使得模型能处理自然语言
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        # 保存模型训练的结果
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        # 记录日志的路径
        self.log_path = dataset + '/log/' + self.model_name
        # dataset + '/data/' + embedding是把这三个合在一起变成一个路径，加载一个.npz文件
        # ["embeddings"]加载的.npz件是一个字典，通常包含多个数组。
        # 这里假设它包含一个名为"embeddings"的数组，表示预训练的词向量。
        # 该数组的形状通常为(词汇表大小, 嵌入维度)，例如(10000,300)，表示有10000个词，每个词的嵌入维度为 300。
        # 数据类型转换.astype('float32')：• 将加载的词向量数组的数据类型转换为 float32
        # if embedding != 'random' else None：• 如果embedding参数不是'random'，则加载预训练的词向量并转换为张量
        # 如果是'random'，则将self.embedding_pretrained设置为None。这意味着如果选择随机初始化词向量，模型将不会使用任何预训练的词向量。
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        # 用GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 2000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4  # 学习率
        # 如果加载了预训练词向量，则用embedding_pretrained.size(1)，否则就是300
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度
        # 每个词嵌入的维度是300，即输入输出大小都是300 eg：词汇表10000，则嵌入层为(10000,300)
        self.dim_model = 300
        # 前馈网络FNN（全连接层）的中间维度
        self.hidden = 1024
        # 最后一层隐藏层，用作更好的把数据分到对应的类中
        self.last_hidden = 512
        # 5头注意力
        self.num_head = 5
        # encoder的层数
        self.num_encoder = 2


'''Attention Is All You Need'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # config就是上边那个类的config，直接传完参数之后搞进来
        if config.embedding_pretrained is not None:  # 使用预训练的词向量
            # nn.Embedding：这是PyTorch中用于创建嵌入层的类。嵌入层的作用是将离散的索引（如词索引）映射到低维的连续向量空间。
            # from_pretrained：这是nn.Embedding的一个类方法，用于从预训练的词向量初始化嵌入层。它接受一个张量（通常是预训练的词向量）作为输入，并将其作为嵌入层的权重。
            # config.embedding_pretrained这是预训练词向量的张量。它通常是一个二维张量，形状为(词汇表大小,嵌入维度)。
            # 如果词汇表大小为 10000，嵌入维度为 300，那么 config.embedding_pretrained的形状为(10000, 300)
            # freeze：这个参数决定是否在训练过程中更新嵌入层的权重。
            # freeze=True：在训练过程中，嵌入层的权重将保持不变，不会更新。
            # freeze=False：在训练过程中，嵌入层的权重会根据训练数据进行更新。
            # 在你的代码中，freeze=False表示预训练的词向量在训练过程中会被更新
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            # 不用预训练的词向量
            # config.n_vocab词表中单词数量
            # config.embed嵌入向量的维度
            # padding_idx=config.n_vocab - 1指定一个特殊的词用于padding填充
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # Positional_Encoding是下边的函数，实现了位置编码【后边去看论文】
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        # encoder搞里头
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        # nn.ModuleList作为一个容器，存储多个模块
        # copy.deepcopy是深拷贝，使得每一个编码器层是独立的，不共享参数
        # 实现的是EncoderLayers
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])  # 这个是创建config.num_encoder个层
        # lastlayer，将前面层提取的特征映射到目标分类标签的维度
        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        # self.fc1 = nn.Linear(config.dim_model, config.num_classes)

    def forward(self, x):  # 预测
        out = self.embedding(x[0])  # 先搞到embedding里
        out = self.postion_embedding(out)  # 再搞上位置编码
        for encoder in self.encoders:  # 搞到encoder层理
            out = encoder(out)
        # 把out除了第一维，其他的都压塌合成一维
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)  # LastLinear
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        # 注意力机制，多头注意力，是下边的一个类
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        # 前馈网络
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):  # 调用Encoder就是调用了这个，进来的先多头注意力再前馈
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):  # 位置掩码
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device  # GPU
        # 这就是位置编码计算公式，也就是SIN(POS/((10000**(I//2)*2)/embed)) OR COS(POS/((10000**(I//2)*2)/embed))
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # 偶数sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # 奇数cos
        self.dropout = nn.Dropout(dropout)  # dropout

    def forward(self, x):  # 调用Positional_Encoding就是搞这个
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


# 缩放点积注意力 通俗易懂的讲就是互相之间的“评价”
class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        # 公式：softmax(Q*K^T/D_K**1/2)*V
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


# 多头自注意力
class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0  # 确保嵌入维度可以被头数整除，因为transformer每个头平均的处理一部分特征，要确保各个头平均处理
        self.dim_head = dim_model // self.num_head  # 将嵌入维度均匀分配到每个头中。
        # 下边这三个FC_QKV,实际上是和嵌入向量做运算的全连接层，只有运算完之后的结果才是真的QKV
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        # 缩放点积注意力
        self.attention = Scaled_Dot_Product_Attention()
        # 前馈
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 层归一化，为何不用批归一化？
        # 首先，我们要进行的是对输入的一个样本进行单独的归一化，而不是整个批次的归一化，这是因为transformer处理的是变长序列，其中有大量的填充
        # 其次，transformer的层次深，使用批归一化可能会使得均值和标准差在不同批次之间不稳定
        # 最后，批归一化保证了每一层输出相似，能够加快训练
        # 内部协变量偏移（Internal Covariate Shift）是指在训练过程中，网络各层的输入分布发生变化，这会影响模型的训练速度和稳定性。

        # Batch Normalization 的局限性•
        # 依赖于批次大小：Batch Normalization 的效果依赖于批次大小，小批次可能导致统计信息不稳定。
        # 对填充敏感：在处理变长序列时，Batch Normalization 会受到填充标记的影响，因为填充标记会引入大量的零，从而误导归一化过程。
        # 不适合自注意力机制：自注意力机制允许模型同时考虑序列的所有位置，而 Batch Normalization 无法有效处理这种全局依赖。
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):  # 调用多头注意力，其实就搞到这里边了
        # 此时，X的形状为(batch_size,sequence_length,dim_model)
        # batch_size为批次大小，sequence_length为每个样本经过填充or截断之后的最大长度，dim_model为嵌入向量的维度
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        # 这玩意就是根号下DK
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        # 将多头注意力出来的形状(batch_size,num_head,sequence_length,num_head)改为原来的形状，即
        # (batch_size,sequence_length,num_head*dim_head)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        # 前馈
        out = self.fc(context)
        # dropout
        out = self.dropout(out)
        # 残差
        out = out + x  # 残差连接
        # 层归一化
        out = self.layer_norm(out)
        return out


# FNN，位置前馈网络
class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        # 全连接层1和2
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)  # dropout
        self.layer_norm = nn.LayerNorm(dim_model)  # 层归一化

    def forward(self, x):
        # X是(batch_size,sequence_length,dim_model)
        out = self.fc1(x)
        # 对隐藏层特征应用ReLU激活函数，引入非线性。
        # self.fc2(out)：将隐藏层特征通过第二个全连接层，映射回原始维度。
        # self.dropout(out)：应 Dropout，防止过拟合。
        # out+x：残差连接，将变换后的特征与输入特征相加。
        # self.layer_norm(out)：层归一化，对每个样本的特征进行归一化，稳定训练过程。
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
