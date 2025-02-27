import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# 这个是老的写法，整个过程是
# data=torch.Tensor[1.0],
# w=Variable(data,requires_grad=True)
# w = Variable(torch.Tensor([1.0]), requires_grad=True)  # Any random value

# 现代的写法
w = torch.tensor([1.0], requires_grad=True)


# our model forward pass


def forward(x):
    return x * w


# Loss function


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# Before training
print("predict (before training)", 4, forward(4).data[0])

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        # 现在不推荐这么写了
        # w.data = w.data - 0.01 * w.grad.data
        # Manually zero the gradients after updating weights
        # w.grad.data.zero_()

        #其实这个就是手动实现的optimizer（）
        with torch.no_grad():
            w -= 0.01 * w.grad

        # 清零梯度
        w.grad.zero_()

    print("progress:", epoch, l.item())

# After training
print("predict (after training)", 4, forward(4).data[0])
