# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


# 构建词表
# file_path文本路径
# tokenizer分词器函数
# max_size词表最大大小
# min_freq单词最小出现频率
def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    # tqdm是显示进度条的
    with open(file_path, 'r', encoding='UTF-8') as f:  # 读入
        for line in tqdm(f):
            lin = line.strip()  # 去除空格
            if not lin:  # 去除空行
                continue
            content = lin.split('\t')[0]  # 用\t来切割，用第一部分
            for word in tokenizer(content):  # 搞到分词器里
                vocab_dic[word] = vocab_dic.get(word, 0) + 1  # 遍历每个单词，统计其出现频率并存储在vocab_dic中。
        # [_ for _ in vocab_dic.items() if _[1] >= min_freq]返回一个元组，出现频率大于min_freq的
        # sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)
        # 进行排序，排序规则为按照第二个元素也就是x[1]排序，然后是降序，返回一个元组
        # [:max_size]拿到前max_size个，返回一个列表，里边是(word,count)样式的元组
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]
        # enumerate(vocab_list)遍历 vocab_list，同时返回每个元素的索引idx和内容word_count
        # word_count[0]：每个元组的第一个元素，即单词
        # idx：单词的索引，从0开始。
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        # 作用：将UNK和PAD添加到词汇表字典中，并分配唯一的索引
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:  # 如果使用字输入
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:  # eg：hello 变成h e l l o
        tokenizer = lambda x: [y for y in x]  # 字符级别的输入，char-level
    if os.path.exists(config.vocab_path):  # 词表路径存在
        vocab = pkl.load(open(config.vocab_path, 'rb'))  # 读出来
    else:
        # 词表路径不存在，就自己构建词表，用上边类构造
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        # 将构建好的词汇表保存到文件中
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")  # 打印个词表大小

    # 这个在def里def是叫做闭包
    # 闭包可以访问其外部函数的变量，即使外部函数已经执行完毕。
    # 自由变量：闭包可以访问其外部函数的变量，这些变量称为自由变量。
    # 持久性：即使外部函数已经执行完毕，闭包仍然可以访问其外部函数的变量。
    # 封装性：闭包可以封装外部函数的变量，使其对外部不可见。
    def load_dataset(path, pad_size=32):  # 读取数据
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):  # 限时进度条
                lin = line.strip()  # 去空格
                if not lin:  # 空行跳
                    continue
                content, label = lin.split('\t')  # \t划分开
                words_line = []  # 准备存
                token = tokenizer(content)  # 搞到tokenizer里边进行分词
                seq_len = len(token)  # 计算序列的实际长度。
                if pad_size:  # 如果指定了pad_size
                    if len(token) < pad_size:  # 如果长度小，那么久填充
                        token.extend([PAD] * (pad_size - len(token)))
                    else:  # 否则就截断，拿前pad_size个
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:  # 单独把词拿出来
                    # 将单词映射为索引，如果单词不在词汇表中，则使用UNK的索引。
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                # 将处理后的数据添加到contents中
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    # 读取各种数据
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


# 自定义的数据迭代器，用于在训练过程中按批次加载和处理数据
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        # 一个包含所有数据的列表，每个元素是一个元组(x, y, seq_len)，其中x是单词索引列表,y是标签.seq_len是序列的实际长度。
        self.batch_size = batch_size
        self.batches = batches  # 批次
        self.n_batches = len(batches) // batch_size  # 算出有多少个批次
        self.residue = False  # 记录数据是否可以被batch_size整除。如果不能整除，最后一个批次会包含少于batch_size的样本。
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0  # 当前索引批次，初始化为0
        self.device = device  # 设备，GPU OR CPU

    # 如果说使用的是DataLoader，就可以不用转化为张量，省略这一步[最新pytorch]
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)。搞到设备上
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:  # 如果数据可以背batch_size整除并且已经到了最后一个批次
            batches = self.batches[self.index * self.batch_size: len(self.batches)]  # 提取剩余的数据
            self.index += 1  # 索引加一，表示已经处理了剩余数据
            batches = self._to_tensor(batches)  # 剩余转化为张量
            # 返回一个元组((x, seq_len), y)
            # 其中x和seq_len是输入数据。y是标签
            return batches

        elif self.index >= self.n_batches:  # 如果已经搞完了
            self.index = 0  # 方便下次运行
            raise StopIteration  # raise StopIteration：抛出StopIteration异常，表示迭代结束。
        else:  # 没搞完且不是最后一个就正常搞
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):  # 这是 Python 迭代器协议的一部分，用于返回迭代器对象本身
        return self

    def __len__(self):  # 这是 Python 迭代器协议的一部分，用于返回迭代器的长度。
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):  # 构建迭代器，把所需的数据通过config传进去
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300  # 词向量的维度。
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"  # 提取词向量后的保存位置
    if os.path.exists(vocab_dir):  # 如果词汇表存在
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:  # 不存在就构建词汇表
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        # 构建词表，build_vocab是上边的函数
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))
    # 初始化一个随机的词向量矩阵，形状为(词汇表大小,词向量维度)
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')  # 打开预训练的词向量
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")  # 读取每一行，用空格划开
        if lin[0] in word_to_id:  # 如果这个词已经在word_to_id里了【word_to_id是一个字典】
            idx = word_to_id[lin[0]]  # 获取当前单词在词汇表中的索引，以便将词向量存储到正确的位置。
            # lin[1:301]：当前行从第二个元素到第301个元素，表示词向量的各个维度
            # float(x)：将词向量的每个维度转换为浮点数。
            # 作用：提取当前单词的词向量，并将其转换为浮点数列表。
            emb = [float(x) for x in lin[1:301]]
            # embeddings：初始化的词向量矩阵，形状为(词汇表大小,词向量维度)
            # np.asarray(emb,dtype='float32')：将提取的词向量转换为NumPy数组，并指定数据类型为float32
            # 作用：将提取的词向量存储到词向量矩阵中，索引为idx
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()  # 关掉f
    # 将提取的词向量矩阵保存到文件中，便于后续加载和使用。
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
