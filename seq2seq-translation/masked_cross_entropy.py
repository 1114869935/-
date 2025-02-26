import torch
from torch.nn import functional
from torch.autograd import Variable


# 用于实现交叉熵损失函数即：-sum（1/N*ylog（y））
# 生成一个掩码，用于表示序列在哪些地方是无效的
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()  # 如果没有指定最大长度，就以给的最大长度来计
    batch_size = sequence_length.size(0)  # 获取第一个张量的维度，也就是batch_size
    seq_range = torch.arange(0, max_len - 1).long()  # 用于生成一个大小为max_len的序列，然后转化为long的类型
    # seq_range.unsqueeze(0)，会将第0维扩展成1，然后expand又将第0维的1扩展为batch_size
    # 为啥呢，因为要和每个样本批次进来的维度对应一致。即(batch_size, max_len)，这样才能有掩码的作用
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    # 这里之前已经实战过了，啥意思呢，这个Variable会讲把张量包装为自动求导的变量，但现在直接用张量即可
    seq_range_expand = Variable(seq_range_expand)
    # 如果 sequence_length已经在GPU上，就把 seq_range_expand也搞上去
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    # 把sequence_length从(batch_size,)变成(batch_size,1)，然后在变成(batch_size，maxlen)
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    # • 比较   seq_range_expand   和   seq_length_expand  ，生成一个布尔掩码。• 形状为   (batch_size, max_len)  ，
    # 其中有效位置为   True  （序列长度范围内），无效位置为   False  （超出序列长度的部分）。
    return seq_range_expand < seq_length_expand


# 实现了 掩码交叉熵损失 公式：sum(mask*LOSS)/sum(mask),sum是把矩阵的每一行都加在一起，这样就变成了数字除以数字了
def masked_cross_entropy(logits, target, length):
    # 这个原来是Variable，现在可以不用了，直接可以往里边写了
    length = torch.LongTensor(length).cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    #扁平化logits的原始形状是(batch_size, max_len, num_classes)#num_classes是分类问题能分到多少类，即有多少个单词
    # 使用 view(-1, logits.size(-1))   将   logits   扁平化为   (batch_size * max_len, num_classes)  。
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    #计算每个类别的对数概率
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    #  target   的原始形状是   (batch_size, max_len)
    #  。• 使用   view(-1, 1)   将   target   扁平化为   (batch_size * max_len, 1)  。
    #  • 这一步是为了将所有时间步的真实标签合并到一个二维张量中，便于后续计算损失。
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    # 使用   torch.gather   根据   target_flat   索引获取每个时间步的真实标签对应的对数概率。
    # •   torch.gather   的作用是从   log_probs_flat   中取出每个样本的真实标签对应的概率值
    # • 计算负对数似然损失（Negative Log-Likelihood, NLL）：
    # •   losses_flat   的形状为   (batch_size * max_len, 1) ，表示每个时间步的损失。
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    #恢复原状
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    #生成掩码矩阵
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss
