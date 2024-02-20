import torch
from torch import nn
import torchvision
from PIL import Image as imim  #tkinter中也含有Image类
from PIL import ImageTk
from tkinter import *
import tkinter as tk


#显示图片函数
def show():
    clear_text()
    img_loc=eval(entry1_1.get()) #图片路径
    img=imim.open(img_loc).resize((200,200))
    global photo1
    photo1=ImageTk.PhotoImage(img)
    tk.Label(window,image=photo1).place(x=500,y=50)

#创建界面
window=tk.Tk()
window.title('图片识别')
window.geometry('1000x600')
window.configure(bg='black')

#标题标签
title=tk.Label(window,text='图片识别器',bg='black',font=('Arial',30),fg='white',width=20,height=2)#fg设置字体颜色
title.place(x=10,y=10)


#输入参数标签+文本框
label1=tk.Label(window,text='图片路径',bg='black',font=('Arial',20),fg='yellow',width=10,height=1)
label1.place(x=30,y=130)
label2=tk.Label(window,text='识别结果',bg='black',font=('Arial',20),fg='yellow',width=10,height=1)
label2.place(x=30,y=200)
entry1_1=tk.Entry(window,width=30)
entry1_1.place(x=180,y=140)
text1_1=tk.Text(window,width=30,height=2)
text1_1.place(x=180,y=210)


# img_path='image1.jpg'
# image=imim.open(img_path)
# # print(image)
# image=image.convert('RGB')
#
# transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),#这里Resize是将图片的尺寸转为32*32，因为之前训练的网络对图片的要求就是32*32
#                                           torchvision.transforms.ToTensor()])
# image=transform(image)
# print(image.size())


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

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

#识别函数
def recognize():
    clear_text()
    var=eval(entry1_1.get())
    image = imim.open(var)
    # print(image)
    image = image.convert('RGB')

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32, 32)),  # 这里Resize是将图片的尺寸转为32*32，因为之前训练的网络对图片的要求就是32*32
         torchvision.transforms.ToTensor()])
    image = transform(image)

    model = torch.load("mxc_cpu(version3)38.pth")  # 加载网络模型
    # print(model)

    image = torch.reshape(image, (1, 3, 32, 32))  # 需要的是4维数据，但图片是3维，用这行代码进行改写
    model.eval()
    with torch.no_grad():
        output = model(image)
    # print(output)

    # print(output.argmax(1))#这里输出的tensor[]里的序号就是对完整模型训练套路第11行处debug后打开classes看到的序号对应的东西

    result = str(output.argmax(1))
    code = result.split('[')[-1].split(']')[0]

    if code == '0':
        text1_1.insert('insert','飞机')
    elif code == '1':
        text1_1.insert('insert','汽车')
    elif code == '2':
        text1_1.insert('insert','鸟')
    elif code == '3':
        text1_1.insert('insert','猫')
    elif code == '4':
        text1_1.insert('insert','鹿')
    elif code == '5':
        text1_1.insert('insert','狗')
    elif code == '6':
        text1_1.insert('insert','青蛙')
    elif code == '7':
        text1_1.insert('insert','马')
    elif code == '8':
        text1_1.insert('insert','轮船')
    elif code == '9':
        text1_1.insert('insert','卡车')

# print(code)

# if code=='0':
#     print('飞机')
# elif code=='1':
#     print('汽车')
# elif code=='2':
#     print('鸟')
# elif code=='3':
#     print('猫')
# elif code=='4':
#     print('鹿')
# elif code=='5':
#     print('狗')
# elif code=='6':
#     print('青蛙')
# elif code=='7':
#     print('马')
# elif code=='8':
#     print('轮船')
# elif code=='9':
#     print('卡车')

def clear_text():
    #注意，这里要想NORMAL和DISABLE不报错，需要加上from tkinter import *
    # 开启编辑text
    text1_1.config(state=NORMAL)
    text1_1.delete("1.0", "end")
    # entry1_1.config(state=NORMAL)
    # entry1_1.delete(0,END)

#清空函数
def clear():
    #注意，这里要想NORMAL和DISABLE不报错，需要加上from tkinter import *
    # 开启编辑text
    text1_1.config(state=NORMAL)
    text1_1.delete("1.0", "end")
    entry1_1.config(state=NORMAL)
    entry1_1.delete(0,END)

#按钮
button1=tk.Button(window,text='显示',bg='grey',font=('Arial',12),width=10,height=2,command=show)
button1.place(x=100,y=500)
button2=tk.Button(window,text='识别',bg='grey',font=('Arial',12),width=10,height=2,command=recognize)
button2.place(x=300,y=500)
button3=tk.Button(window,text='清空',bg='grey',font=('Arial',12),width=10,height=2,command=clear)
button3.place(x=500,y=500)

window.mainloop()


# import torch.nn as nn
# import torch.nn.functional as F
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))