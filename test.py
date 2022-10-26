import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from CGAN import *
import scipy.io as io
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_dictG", default="model/netG_epoch1_30_car_1800.pth")#选择生成器的权重
parser.add_argument("--img", default="/home/ps/DiskC/lyf-data/matlab_gan1/data/30_hxy2/radar_heatmap_9_255.mat")#选择需要测试的点云图像
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Generator().to(device)
# net = nn.DataParallel(net )
net.load_state_dict(torch.load(args.pretrained_dictG))


def imshow(path):
    """展示结果"""
    pilImg = io.loadmat(path)
    img = torch.tensor(pilImg['radar_heatmap']).unsqueeze(0).float().permute(0,2,3,1).unsqueeze(0).to(device)
    source = net(img)[0, :, :,:]
    source = source.cpu().detach().numpy()  # 转为numpy
    source = source.transpose((1, 2, 0))  # 切换形状
    source = np.clip(source, 0, 1)  # 修正图片
    print(source.shape )
    # fakeImgs = net(img ).detach().cpu().numpy()[0][0]
    #
    # plt.imshow(fakeImgs)
    #
    # plt.show()
    # plt.pause(0.01)
    plt.imshow(source)
    img = Image.fromarray(np.uint8(source[:,:,0]* 255))

    img.save('./result/' +path.split('/')[-1][:-4] + '_result.jpg')  # 将数组保存为图片，保存图片的路径

imshow(args.img)
