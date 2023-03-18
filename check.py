import os
import json
 
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
 
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    data_transform = transforms.Compose(    # 定义图片预处理函数，用来对载入图片进行预处理操作
        [transforms.Resize((32, 32)),   # 缩放到224*224
         transforms.ToTensor(),   # 转化为一个tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    # 标准化处理
    txtfilepath = r"F:\JetBrains\cv任务\cv\CV\check\img"   # 原始txt文件所存文件夹，文件夹可以有一个或多个txt文件
    total_txt = os.listdir(txtfilepath)   # 返回指定的文件夹包含的文件或文件夹的名字的列表
    num = len(total_txt)
    list = range(num)  # 创建从0到num的整数列表
 
    for i in list:
        name = total_txt[i]
        # load image
        print(txtfilepath+"\\"+name)
        img = Image.open(txtfilepath+"\\"+name, 'r') #读取文件
        
        plt.imshow(img)  # 展示输入的图片
        #input()
        #continue
        # [N, C, H, W]
        img = data_transform(img)   # 调用预处理函数，对载入读片进行预处理
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)  # 预处理之后扩充一个维度（batch维度），这与Alexnet输入有关（具体见NB笔记）
 
        # read class_indict
        json_path = './class_indices.json'   # 读取保存的json文件（类别名称以及对应的索引）
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
 
        json_file = open(json_path, "r")  # 解码成所需要的字典
        class_indict = json.load(json_file)
 
        # create model
        model = AlexNet(num_classes=10).to(device)   # 初始化网络
 
        # load model weights
        weights_path = "./alex_animal.pth"
        #print(os.path.abspath(weights_path))
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))  # 载入网络模型
 
        model.eval()   # 进入eval模式（即关闭掉droout方法）
        with torch.no_grad():   # 不跟踪变量的损失梯度
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()   # 将图片通过model正向传播，得到输出，将输入进行压缩，将batch维度压缩掉，得到最终输出（out）
            predict = torch.softmax(output, dim=0)  # 经过softmax处理后，就变成概率分布的形式了
            predict_cla = torch.argmax(predict).numpy()  # 通过argmax方法，得到概率最大的处所对应的索引
            #print(predict_cla)
            #print("------")
 
 
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())   # 打印类别名称以及他所对应的预测概率
        print(print_res+'\n')
        plt.title(print_res)
        for i in range(len(predict)):
            with open('test.txt', 'a') as file0:  # 将以下print内容保存到test.txt文件中
                print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy()), file=file0)
        with open('test.txt', 'a') as file0:
            print("--------------------------我是可爱的分隔线--------------------------", file=file0)
        # plt.show()
 
 
if __name__ == '__main__':
    main()