# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

# python train.py --model TextCNN --embedding random --word True
# 通过上边命令来运行，意思就是说model是TextCNN,embedding是random,word是True
# 下边的这一堆是定义一个命令行参数，然后告诉程序如何解析用户在命令行之中输入的参数，从而在程序中使用
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    # 以下的都是参数读取和设置
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'  # 随机拿一个数据集
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    # 以下的处理是因为FastText的数据集需要额外的处理，如果看不完就不看了
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif

        embedding = 'random'
    else:
        # 别的都用一种数据
        from utils import build_dataset, build_iterator, get_time_dif
    # import_module是Python的内置函数，来自importlib模块，用于动态导入模块
    # 'models.' + model_name 是模块的路径。假设model_name是'Transformer'，那么实际导入的模块路径是'models.Transformer'
    x = import_module('models.' + model_name)
    # Config就是所对应模块中的config，把这几个参数传进去开始搞
    config = x.Config(dataset, embedding)

    # 以下的这几行代码用处是保证你的模型每次运行时都能产生相同的结果
    # np.random.seed(1)代表随机种子是1，只要随机种子相同，那么每次随机的时候生成的随机数序列就是相同的
    np.random.seed(1)
    # torch.manual_seed(1)确保PyTorch生成的随机数序列在每次运行时都相同。
    torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)确保在多GPU环境中，每个GPU生成的随机数序列在每次运行时都相同。
    torch.cuda.manual_seed_all(1)
    # cuDNN是NVIDIA 提供的深度神经网络库，用于加速GPU上的深度学习计算。
    # 某些操作（如卷积和池化）在默认情况下是非确定性的，这意味着即使输入相同，输出也可能不同。
    # 设置 torch.backends.cudnn.deterministic = True可以确保这些操作的行为是确定性的。
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # 记录开始时间
    start_time = time.time()
    print("Loading data...")
    # build_dataset是utils里的一个模块，为构建数据集
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    # 构建训练集迭代器
    train_iter = build_iterator(train_data, config)
    # 构建验证集迭代器
    dev_iter = build_iterator(dev_data, config)
    # 构建测试集迭代器
    test_iter = build_iterator(test_data, config)
    # 记录运行代码用的时间
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    # vocab是一个字典，key是词，value是出现次数，这样可以拿到有多少个词
    config.n_vocab = len(vocab)
    # 将配置参数传递给模型类，用于初始化模型，然后搞到device上
    model = x.Model(config).to(config.device)
    # 如果不是transformer模型
    if model_name != 'Transformer':
        # 预训练权重：在许多情况下，Transformer模型会使用预训练的权重（如BERT、GPT等）
        # 这些预训练的权重已经通过大规模数据训练得到，通常不需要重新初始化
        # 初始化策略：如果Transformer模型没有使用预训练权重，通常会使用特定的初始化策略（如Xavier或Kaiming初始化）
        # 这些策略已经嵌入到模型的定义中，因此不需要额外的初始化函数
        init_network(model)  # 就需要初始化模型的权重
    print(model.parameters)#打印参数
    train(config, model, train_iter, dev_iter, test_iter)#训练
