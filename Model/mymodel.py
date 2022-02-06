import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from d2l import torch as d2l
from torch.utils.data import Dataset
from torchvision import transforms
import csv
from PIL import Image
from torch.utils.data import DataLoader
import os, glob
import random




def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


# 训练主函数
def train(train_iter, test_iter, net, loss, optimizer, num_epochs, batch_size,mode='0'):
    # 初始化训练细节打印间隔
    log_interval = 20
    # 存储可视化图表数据
    batch_num_in_one_epoch = len(train_iter.dataset) / batch_size
    test_acc = []
    train_acc = []
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []
    # 初始化网络参数，并放在GPU上训练
    if mode != 'continue':
        net.apply(init_weights)
    device = d2l.try_gpu()
    net.to(device)
    print("training on", device)
    net.train()
    # 训练循环
    for epoch in range(num_epochs):
        # 训练前测试一次
        if epoch == 0:
            test_losses_item, test_acc_item = test(test_iter, net, loss, device)
            test_losses.append(test_losses_item)
            test_acc.append(test_acc_item)
            test_counter.append(epoch)
        # 小批量循环
        for i, (X, y) in enumerate(train_iter):
            train_correct = 0
            train_pred = 0
            optimizer.zero_grad()  # 清空梯度
            X, y = X.to(device), y.to(device)  # 将数据放到GPU上计算
            y_hat = net(X)
            train_pred = y_hat.data.max(1, keepdim=True)[1]
            train_correct += train_pred.eq(y.data.view_as(train_pred)).sum()
            l = loss(y_hat, y)  # 计算损失函数
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
            train_acc.append(train_correct/batch_size)
        # 每个周期后测试一次
        test_losses_item, test_acc_item = test(test_iter, net, loss, device)
        test_losses.append(test_losses_item)
        test_acc.append(test_acc_item)
        test_counter.append(epoch + 1)
        # 每个周期后保存一次
        torch.save(net.state_dict(), "./Parameters/model.pth")

    # 可视化
    details(train_counter, train_losses, train_acc, test_counter, test_losses, test_acc)
    

def details(
    train_counter, train_losses, train_acc, test_counter, test_losses, test_acc
):
    fig, ax = plt.subplots(1, 2)
    ax2 = ax[0].twinx()  # 同一坐标系下创建第二个竖坐标轴

    ax[0].plot(train_counter, train_losses, color="blue", label="train loss")
    ax2.bar(test_counter, test_losses, alpha=0.3, color="red", label="test loss")
    
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("Train Loss")
    ax2.set_ylabel("Test Loss")

    ax[1].plot(train_counter, train_acc, color="purple", label="train acc",linestyle='--')
    ax[1].plot(test_counter, test_acc, color="green", label="test acc",linestyle='--')
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("Accuracy")

    
    fig.legend(loc = 'upper center')
    plt.subplots_adjust(left=0.09,bottom=0.098,right=0.936,top=0.774,wspace=0,hspace=0.271)
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
    test_acc = correct / len(test_iter.dataset)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_iter.dataset),
            100.0 * test_acc,
        )
    )
    return test_loss.item(), test_acc.item()


def make_csv(filename="myself_data.csv", root=None):
    work_path = os.getcwd()
    if root == None:
        raise ValueError("Please enter root!")
    if filename[-4:] != ".csv":
        raise ValueError("Please check filename!")

    class_to_num = {}
    class_name_list = os.listdir(root)
    for class_name in class_name_list:
        class_to_num[class_name] = len(class_to_num.keys())  # 所有类别的名字按顺序分配序号
    image_dir = []
    for class_name in class_name_list:  # 得到每个样本的路径
        image_dir += glob.glob(os.path.join(root, class_name, "*.png"))
    random.shuffle(image_dir)
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        for image in image_dir:
            class_name = image.split(os.sep)[-2]  # 拿到路径里的类名，再把类名转换为之前定义好的序号，即为该图像的标签
            label = class_to_num[class_name]
            writer.writerow([image, label])
    print(
        f"Csv_File has been made from [{root}] and saved as",
        f"['{filename}']",
        "in",
        f"[{work_path}]",
    )
    for classname in class_to_num:
        print(classname, "-->", class_to_num[classname])

    # 示例
    # make_csv('1.csv',r'C:\Users\GLT\Desktop\Nums')
    return filename


class makedataset(Dataset):
    def __init__(self, csv_filename, resize, mode):
        super(makedataset, self).__init__()
        self.csv_filename = csv_filename
        self.resize = resize
        self.image, self.label = self.load_csv()

        if mode == "train":
            self.image = self.image[: int(0.6 * len(self.image))]
            self.label = self.label[: int(0.6 * len(self.label))]
        elif mode == "test":
            self.image = self.image[
                int(0.6 * len(self.image)) : int(0.8 * len(self.image))
            ]
            self.label = self.label[
                int(0.6 * len(self.label)) : int(0.8 * len(self.label))
            ]
        else:
            self.image = self.image[int(0.8 * len(self.image)) :]
            self.label = self.label[int(0.8 * len(self.label)) :]

    def load_csv(self):
        image, label = [], []
        with open(self.csv_filename) as f:
            reader = csv.reader(f)
            for row in reader:
                i, l = row
                image.append(i)
                label.append(int(l))
        return image, label

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        tf = transforms.Compose(
            [
                lambda x: Image.open(x).convert("RGB"),
                transforms.Resize(self.resize),
                transforms.ToTensor(),
            ]
        )
        image_tensor = tf(self.image[idx])
        label_tensor = torch.tensor(self.label[idx])
        return image_tensor, label_tensor


def get_iter(csv_filename, resize=False, mode=None, batch_size=None):
    dataset = makedataset(csv_filename, resize, mode)
    iter = DataLoader(dataset, batch_size)
    print("Dataset Examples:")
    for x, y in iter:
        print(x.shape)
        print("label:", y, "\n")
    return iter

def predict(net, test_iter, n=6):  # @save
    device = d2l.try_gpu()
    net.to(device)
    net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            X = X[0:n]
            y = y[0:n]
            X, y = X.to(device), y.to(device)
            trues = get_classname_from_label(y)
            preds = get_classname_from_label(net(X).argmax(axis=1))
            titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
            print(titles)
            d2l.show_images(X.cpu().reshape((n, 28, 28)), 1, n, titles=titles[0:n])
            plt.subplots_adjust(top = 0.7)
            plt.show()

def get_classname_from_label(labels):
    text_labels = ['0','1','2','3','4','5','6','7','8','9']
    return [text_labels[int(i)] for i in labels]