
import torch
from torch.autograd import Variable
import torch.nn.functional as F

#x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
#y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))

x_data=torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data=torch.tensor([[0.], [0.], [1.], [1.]])
class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        """
        super(Model, self).__init__()
        #对于逻辑回归模型，最基本的函数，也就是Z函数，依然是w*x+b
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        #Z函数套壳一个G函数，即sigmoid函数，这个也叫做激活函数，也还有另一种写法，见07
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.


#BCELoss是二元交叉熵损失函数，用于二分类问题
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
        # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# After training
hour_var = Variable(torch.Tensor([[1.0]]))
print("predict 1 hour ", 1.0, model(hour_var).data[0][0] > 0.5)
hour_var = Variable(torch.Tensor([[7.0]]))
print("predict 7 hours", 7.0, model(hour_var).data[0][0] > 0.5)
