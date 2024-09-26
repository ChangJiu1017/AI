# import torch
# print(torch.__version__)  # 打印 PyTorch 的版本
# print(torch.cuda.is_available())  # 检查 CUDA 是否可用（若支持 GPU）

#
# import torch
#
# # 创建一个 5x3 的随机矩阵
# x = torch.rand(5, 3)
# print("随机矩阵 x:")
# print(x)
#
# # 检查是否有可用的 GPU
# if torch.cuda.is_available():
#     device = torch.device("cuda")  # 使用 GPU
#     y = torch.ones_like(x, device=device)  # 将矩阵 y 创建在 GPU 上
#     x = x.to(device)  # 将矩阵 x 移动到 GPU
#     z = x + y
#     print("矩阵 z (使用 GPU):")
#     print(z)
# else:
#     print("没有可用的 GPU，使用 CPU 进行计算。")
#
# # 验证结果
# print("程序运行成功!")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 4)  # 输入为2个特征，输出4个特征
        self.layer2 = nn.Linear(4, 1)  # 输出1个值（二分类）

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # 使用ReLU激活函数
        x = torch.sigmoid(self.layer2(x))  # 使用Sigmoid激活函数
        return x

# 生成随机训练数据
# 我们将生成100个样本，每个样本有两个特征
torch.manual_seed(0)  # 设置随机种子
data = torch.randn(100, 2)  # 100个样本，每个样本有2个特征
labels = (data[:, 0] + data[:, 1] > 0).float().unsqueeze(1)  # 简单的二分类标签

# 定义模型、损失函数和优化器
model = SimpleNN()
criterion = nn.BCELoss()  # 使用二分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 开始训练
losses = []
for epoch in range(100):  # 训练100个epoch
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 绘制损失变化曲线
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

