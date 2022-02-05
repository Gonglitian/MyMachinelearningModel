import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from d2l import torch as d2l

# 训练主函数
def train(train_iter, test_iter, net, loss, optimizer, num_epochs, batch_size):
    # 初始化训练细节打印间隔
    log_interval = 20
    # 存储可视化图表数据
    # plt.ion()#动态绘制图像
    batch_num_in_one_epoch = len(train_iter.dataset) / batch_size
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []
    # 定义Xavier初始化函数
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # 初始化网络参数，并放在GPU上训练
    net.apply(init_weights)
    device = d2l.try_gpu()
    net.to(device)
    print("training on", device)
    net.train()
    # 训练循环
    for epoch in range(num_epochs):
        # 训练前测试一次
        if epoch == 0:
            test_losses.append(test(test_iter, net, loss, device))
            test_counter.append(epoch)
        # 小批量循环
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()  # 清空梯度
            X, y = X.to(device), y.to(device)  # 将数据放到GPU上计算
            l = loss(net(X), y)  # 计算损失函数
            l.backward()  # 损失函数反向传播
            optimizer.step()  # 更新梯度
            # 训练细节
            if i % log_interval == 0:  # 训练完一定数量的小批量后打印训练细节：周期、进度、损失值
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch + 1,
                        i * len(X),
                        len(train_iter.dataset),
                        100.0 * i / len(train_iter),
                        l,
                    )
                )
            # 每个小批量后记录训练损失
            train_counter.append(i / batch_num_in_one_epoch + epoch)
            train_losses.append(l.item())
        # 每个周期后测试一次
        test_losses.append(test(test_iter, net, loss, device))
        test_counter.append(epoch+1)
        # 每个周期后保存一次
        torch.save(net.state_dict(), "./Parameters/model.pth")
        torch.save(optimizer.state_dict(), "./Parameters/optimizer.pth")

    #可视化
    fig, ax1 = plt.subplots(1, 1)
    ax2 = ax1.twinx()  # 同一坐标系下创建第二个竖坐标轴

    ax1.plot(train_counter, train_losses, color="blue", label="train loss")
    ax2.bar(test_counter, test_losses, alpha=0.1, color="red", label="test loss")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Train Loss")
    ax2.set_ylabel("Test Loss")

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.show()
    # plt.pause(2)
    # plt.close()


def test(test_iter, net, loss, device):
    print("Waiting for test result...")
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            test_loss += loss(y_hat, y)
            pred = y_hat.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()
    test_loss /= len(test_iter.dataset)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_iter.dataset),
            100.0 * correct / len(test_iter.dataset),
        )
    )
    return test_loss.item()
