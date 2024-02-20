#CPU训练版，速度较慢
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

def plot_curve(data):
    plt.plot(range(len(data)),data)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Loss Tendency')
    plt.show()

#准备数据集
#训练数据集
train_data=torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)
#测试数据集
test_data=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

# print(type(train_data))

#获取2个数据集的长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度为{}".format(train_data_size))
print("测试数据集的长度为{}".format(test_data_size))

#利用dataloader来加载数据集
train_data_loader=DataLoader(train_data,batch_size=64)
test_data_loader=DataLoader(test_data,batch_size=64)

#因为这个版本的nn没有Flatten展平这个类，所以得自己写，在CSDN上招的
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

#搭建神经网络
#in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2(padding：是否会在图片旁边进行填充)
class MXC(nn.Module):
    def __init__(self):
        super(MXC, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x=self.model(x)
        return x

#创建网络模型
mxc=MXC()

#创建损失函数
loss_fn=nn.CrossEntropyLoss()

#优化器
learning_rate=1e-2
optimizer=torch.optim.SGD(params=mxc.parameters(),lr=learning_rate)#param是网络模型，模型lr是学习速率

#设置训练网络的一些参数
#记录训练的次数
total_train_step=0
#记录测试的次数
total_test_step=0
#记录测试的准确率
total_accuracy=0
#训练的轮数
epoch=50

#训练损失
train_loss=[]

#添加tensorboard
writer=SummaryWriter('./logs_train')


for i in range(epoch):
    print("-----第{}轮训练开始------".format(i+1))

    #训练步骤开始
    for data in train_data_loader:
        imgs,targets=data
        output=mxc(imgs)
        loss=loss_fn(output,targets)

        #优化器优化模型
        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播
        optimizer.step()#参数优化

        train_loss.append(loss.item()) #使用.item()精度更高

        total_train_step=total_train_step+1
        if total_train_step%100==0:
            print("训练次数:{} , Loss:{}".format(total_train_step, loss))  # 更正规的可以写成loss.item()
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_data_loader:
            imgs,targets=data
            output=mxc(imgs)
            loss=loss_fn(output,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy=(output.argmax(1)==targets).sum() #方向是横向的，所以argmax(1)里面是1
            total_accuracy=total_accuracy+accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的准确率:{}".format(total_accuracy.float()/test_data_size))
    total_test_step=total_test_step+1#测试的次数，其实就是第几轮
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)

    #保存模型
    torch.save(mxc,"mxc_cpu(version3){}.pth".format(i+1))#mxc_1是cpu版的
    print("模型已保存")
writer.close()
plot_curve(train_loss)