x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value


# our model forward pass


def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient
# gradient是梯度，是LOSS对W求偏导得出来的，如果有多个
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)


# Before training
print("predict (before training)", 4, forward(4))

# 这里符合课上所讲的，把一堆数据分成几个epoch来训练，每次下一个epoch的w由上一个得出，依次传递下去，只不过这里
# 每个epoch都是一样的
# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad  # 0.01是学习率，w_1=w_0-a*gradient
        print("\tgrad: ", x_val, y_val, round(grad, 2))  # round是四舍五入，把grad四舍五入到小数点后2位
        l = loss(x_val, y_val)

    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
print("predict (after training)", "4 hours", forward(4))
