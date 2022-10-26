import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms, datasets
import scipy.io as io
#图像处理操作，包括随机裁剪，转换张量
transform2D = transforms.Compose([
    transforms.Resize((128,256)),
#     transforms.Grayscale(1),
                            transforms.ToTensor(),
                                # transforms.Normalize([0.485, ], [0.229, ])
                                ])

transform3D = transforms.Compose([
    # transforms.Resize((64,32,96)),
                            transforms.ToTensor(),
                                # transforms.Normalize([0.485, ], [0.229, ])
                                ])


class PreprocessDataset(Dataset):
    """预处理数据集类"""

    def __init__(self, imgPath3D='',imgPath2D='', transforms2D=transform2D, transforms3D=transform3D):
        """初始化预处理数据集类"""
        self.transforms2D = transform2D
        self.transforms3D = transform3D
        sorted(imgPath2D)
        sorted(imgPath3D)
        self.imgsfiles =[]
        for _, _, files in os.walk(imgPath3D):
            self.imgs3 = [imgPath3D + file for file in files]


        for i in range(len(files)):

            self.imgsfiles.append(os.path.splitext(files[i])[0])


        np.random.shuffle(self.imgsfiles)  # 随机打乱

        self.imgs2D =[]
        self.imgs3D = []
        for i in range(len(self.imgsfiles)):
            self.imgs3D.append(os.path.join(imgPath3D, self.imgsfiles[i] + '.mat'))
        for i in range(len(self.imgsfiles)):
            self.imgs2D.append(os.path.join(imgPath2D,self.imgsfiles[i]+'.jpg'))



    def __len__(self):
        """获取数据长度"""
        if(len(self.imgs3D)==len(self.imgs2D)):
        #
            return len(self.imgs3D)

        else:

            print("the number of 3d&2d imgs not match")


    def __getitem__(self, index):
        """获取数据"""
        tempImg3D = self.imgs3D[index]

        tempImg3D = io.loadmat(tempImg3D)

        sourceImg3D = torch.tensor(tempImg3D['radar_heatmap']).unsqueeze(0).float().permute(0,2,3,1)
        tempImg2D = self.imgs2D[index]

        tempImg2D = Image.open(tempImg2D)

        sourceImg2D = self.transforms2D(tempImg2D)
        cropImg3D = sourceImg3D


        return cropImg3D, sourceImg2D


