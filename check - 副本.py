import os
import torch
import numpy as np
from PIL import Image
from model import AlexNet
from torchvision import transforms


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# 加载模型
# create model
model = AlexNet(num_classes=10).to(device)   # 初始化网络
weights_path = "alex_animal.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))  # 载入网络模型

#'''
data_transform = transforms.Compose(    # 定义图片预处理函数，用来对载入图片进行预处理操作
        [transforms.Resize((32, 32)),   # 缩放到224*224
         transforms.ToTensor(),   # 转化为一个tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    # 标准化处理
#'''
# 加载测试数据
data_test_x = np.load('data_test_x.npy')

# 转换为Tensor，并且归一化
# data_test_x = torch.from_numpy(data_test_x).float() / 255.0

# 使用模型进行预测
model.eval()
with torch.no_grad():
    for i in range(data_test_x.shape[0]):
        #model.eval()
        image = data_test_x[i].reshape(3, 32, 32)
        image = image.transpose((1, 2, 0))  # 转换通道顺序为 (height, width, channel)
        #image = Image.fromarray(image)
        # [N, C, H, W]
        image = data_transform(image)   # 调用预处理函数，对载入读片进行预处理
        image = torch.unsqueeze(image, dim=0)
        predictions = model(image)
        predicted_classes = predictions.argmax(dim=1)

        # 输出预测结果
        print("i:{}, {}".format(i, predicted_classes))
        #print = "class: {}   prob: {:.3}".format(str(predict_cla),
        #                                         predict[predict_cla].numpy())   # 打印类别名称以及他所对应的预测概率
        continue
    
        #img_data = test_data[i].reshape(32, 32, 3)
        image = data_test_x[i].reshape(3, 32, 32)
        image = image.transpose((1, 2, 0))  # 转换通道顺序为 (height, width, channel)
        # [N, C, H, W]
        image = data_transform(image)   # 调用预处理函数，对载入读片进行预处理
        # expand batch dimension
        image = torch.unsqueeze(image, dim=0)  # 预处理之后扩充一个维度（batch维度），这与Alexnet输入有关（具体见NB笔记）
        #image = image.astype(np.float32) / 255.0  # 将像素值缩放到 [0, 1] 范围内
        output = torch.squeeze(model(image.to(device))).cpu()   # 将图片通过model正向传播，得到输出，将输入进行压缩，将batch维度压缩掉，得到最终输出（out）
        predict = torch.softmax(output, dim=0)  # 经过softmax处理后，就变成概率分布的形式了
        predicted_classes = torch.argmax(predict).numpy()  # 通过argmax方法，得到概率最大的处所对应的索引

        # 取最大预测值所对应的类别
        #predicted_classes = predictions.argmax(dim=1)

        # 输出预测结果
        #print("i:{}, {}".format(i, predicted_classes))
        print = "class: {}   prob: {:.3}".format(str(predict_cla),
                                                 predict[predict_cla].numpy())   # 打印类别名称以及他所对应的预测概率
