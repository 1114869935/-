# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter


# 权重初始化，默认xavier
# method:初始化方法，默认为'xavier'  。
# 支持的初始化方法包括：'xavier'：Xavier初始化，适用于激活函数为tanh的场景
# 'kaiming'：Kaiming初始化，适用于激活函数为ReLU的场景。
# 其他：默认使用标准正态分布初始化。
# exclude除了什么层之外
# seed：随机种子，用于确保初始化的可重复性。
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():  # 依次拿到模型的层名称和参数
        if exclude not in name:  # 名字中不含不需要初始化的，那就初始化
            if 'weight' in name:  # 如果有权重，就初始化
                if method == 'xavier':  # xavier初始化
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':  # kaiming初始化
                    nn.init.kaiming_normal_(w)
                else:  # 如果不用这两个初始化，就用正态分布初始化
                    nn.init.normal_(w)
            elif 'bias' in name:  # 如果是偏置
                nn.init.constant_(w, 0)  # 则初始化为0
            else:  # 只初始化偏置和权重，别的都不初始化
                pass


# 开始训练
def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()  # 记录时间
    model.train()  # 开始训练
    # model.parameters()将模型的所有可训练参数传递给优化器，以便在训练过程中更新这些参数。
    # lr是LearnRate，在config里初始化过，直接用
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # 优化器使用Adam优化器

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # 可以使得学习率调整更为精细，加快收敛，避免震荡，避免过拟合
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    # 初始化验证集的最低损失为正无穷大。
    dev_best_loss = float('inf')
    # 记录上次验证集损失下降的批次编号，用于早停机制（early stopping）
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    # 记录日志的，初始化 TensorBoard 的日志记录器，记录训练过程中的各种指标，便于后续可视化分析。
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):  # 每一周期
        # 打印日志
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):  # 通过训练迭代器，拿到每一个batch的训练数据和标签
            outputs = model(trains)  # 拿到输出
            model.zero_grad()  # 梯度归零
            loss = F.cross_entropy(outputs, labels)  # 计算交叉熵损失
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            if total_batch % 100 == 0:  # 每100个批次输出一次成果
                # 每多少轮输出在训练集和验证集上的效果
                # 将以下三个东西移动到CPU然后计算准确率
                true = labels.data.cpu()  # 标签移动到CPU
                predic = torch.max(outputs.data, 1)[1].cpu()  # 预测得到的拿到CPU
                train_acc = metrics.accuracy_score(true, predic)  # 计算准确率
                # evaluate：一个函数，用于在验证集上评估模型的性能
                # 计算验证集的准确率和损失。
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:  # 如果损失降低了，证明有效
                    dev_best_loss = dev_loss  # 最好的损失改为当前的
                    torch.save(model.state_dict(), config.save_path)  # 保存最佳模型的权重。
                    improve = '*'  # 打印的时候标记有提升
                    last_improve = total_batch  # 将上一次改进的批次记为当次
                else:
                    improve = ''  # 没提升
                # 打印日志
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # writer.add_scalar：将训练和验证的损失和准确率记录到TensorBoard。
                # 作用：记录训练过程中的各种指标，便于后续可视化分析。
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()  # 设置为训练模式
            total_batch += 1  # 训练过的批次加一
            # 验证集loss超过1000batch没下降，结束训练
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:  # 如果是触发了早停机制，那么后续的epoch也就不需要再训练了
            break
    writer.close()  # 关闭 TensorBoard 的日志记录器，释放资源，确保所有日志信息都被正确保存。
    test(config, model, test_iter)  # 去测试


def test(config, model, test_iter):  # 测试
    # test
    # 加载保存的最好的模型
    model.load_state_dict(torch.load(config.save_path))
    # 模型改为评估模式
    model.eval()
    # 记录开始时间
    start_time = time.time()
    # 搞到评估函数里
    # test_acc：测试集上的准确率。
    # test_loss：测试集上的损失。
    # test_report：包含精确率（Precision）、召回率（Recall）和F1分数的报告。
    # test_confusion：混淆矩阵（Confusion Matrix）
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    # 打印结果日志
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    # 计算时间消耗并且打印
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):  # 评估函数
    model.eval()  # 模型改为评估模式
    loss_total = 0  # 总的损失
    predict_all = np.array([], dtype=int)  # 用于存储所有预测的标签。
    labels_all = np.array([], dtype=int)  # 用于存储所有真实的标签。
    with torch.no_grad():  # 禁用梯度计算，因为是评估
        for texts, labels in data_iter:  # 从数据迭代器中拿到文章和类别
            outputs = model(texts)  # 自己预测类别
            loss = F.cross_entropy(outputs, labels)  # 交叉熵损失函数
            loss_total += loss  # 计算总的损失
            labels = labels.data.cpu().numpy()  # 获取他的类别
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()  # 获取模型预测的类别，搞到cpu上
            labels_all = np.append(labels_all, labels)  # 把本身的标签加到存储中
            predict_all = np.append(predict_all, predic)  # 把预测的标签加到存储中

    acc = metrics.accuracy_score(labels_all, predict_all)  # 计算准确率
    if test:  # 如果是测试集，那么计算分类报告和混淆矩阵
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)  # 如果不是测试集，那么就返回准确率和平均损失即可
