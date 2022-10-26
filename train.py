from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
from CGAN import *
import torch.optim as optim
from torchvision.models.vgg import vgg16,vgg11
from dataloader import *
from tqdm import tqdm
import torch.distributed as dist
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import torch.optim.lr_scheduler as lr_scheduler

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6" #所使用的GPU编号，若无多GPU则写 “0”
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0,type=int)
parser.add_argument("--path3D", default='data/train/30_hxy') # 点云图像路径
parser.add_argument("--path2D", default="data/train/30_dxy") # 标签图像路径
parser.add_argument("--batch_size", default=64)
parser.add_argument("--epochs", default=1000)
parser.add_argument("--pretrained_dictG", default=0) # 生成器预训练权重
parser.add_argument("--pretrained_dictD", default=0) # 判别器预训练权重
args = parser.parse_args()


torch.cuda.set_device(args.local_rank)
device=torch.device("cuda", args.local_rank)
torch.distributed.init_process_group(backend="nccl", init_method='env://')# nccl是GPU设备上最快、最推荐的后端
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU: ",torch.cuda.is_available())
#构建数据集

processDataset = PreprocessDataset(imgPath3D = args.path3D,imgPath2D =args.path2D)
trainData = DataLoader(processDataset,batch_size=args.batch_size,num_workers=12,pin_memory=True)

#构造模型

netD = Discriminator().to(device)
netG = Generator().to(device)

if num_gpus > 1:
    netG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG)
    netD = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netD)
    num_gpus = torch.cuda.device_count()

    print('use {} gpus!'.format(num_gpus))
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[args.local_rank],
                                                output_device=args.local_rank)
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[args.local_rank],
                                                output_device=args.local_rank,broadcast_buffers=False)



if args.pretrained_dictG:
    model_dictG=netG.state_dict()
    pretrained_dictG = torch.load(args.pretrained_dictG)
    pretrained_dictG = {k:v for k,v in pretrained_dictG.items() if (k in model_dictG and 'fc' not in k)}
    model_dictG.update(pretrained_dictG)
    netG.load_state_dict(model_dictG)
netG.to(device)

if args.pretrained_dictD:
    model_dictD=netD.state_dict()
    pretrained_dictD = torch.load(args.pretrained_dictD)
    pretrained_dictD = {k:v for k,v in pretrained_dictD.items() if (k in model_dictD and 'fc' not in k)}
    model_dictD.update(pretrained_dictD)
    netD.load_state_dict(model_dictD)
netD.to(device)
#构造迭代器


#构造损失函数
criterion = nn.MSELoss()
criterion1 = nn.BCELoss()
#构造VGG损失中的网络模型
vgg = vgg16(pretrained=True).to(device)

lossNetwork = nn.Sequential(*list(vgg.features)[:31]).eval()

for param in lossNetwork.parameters():
    param.requires_grad = False  #让VGG停止学习

for epoch in range(args.epochs):
    netD.train()
    netG.train()
    all_D_loss = 0.
    all_G_loss = 0.

    if (epoch<=100):
        optimizerG = optim.Adam(netG.parameters(), lr=0.0001)

        schedulerG = lr_scheduler.ExponentialLR(optimizerG, gamma=0.999)
        optimizerD = optim.Adam(netD.parameters(), lr=0.0001)
        schedulerD= lr_scheduler.ExponentialLR(optimizerG, gamma=0.999)


    else:
        optimizerG = optim.Adam(netG.parameters(), lr=0.00001)

        schedulerG = lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)
        optimizerD = optim.Adam(netD.parameters(), lr=0.00001)
        schedulerD= lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)
    processBar = tqdm(enumerate(trainData, 1))

    for i, (cropImg3D, sourceImg2D) in processBar:

        cropImg3D, sourceImg2D = cropImg3D.to(device), sourceImg2D.to(device)
        real_outputs = netD(cropImg3D, sourceImg2D)
        fake_img = netG(cropImg3D)  # Generate fake images
        fake_outputs = netD(cropImg3D.detach(), fake_img.detach())
        real_labels = torch.ones_like(real_outputs, dtype=torch.float)
        fake_labels = torch.zeros_like(fake_outputs , dtype=torch.float)
        D_real_loss = criterion1(real_outputs, real_labels)
        D_fake_loss = criterion1(fake_outputs, fake_labels)

        D_loss = D_real_loss + D_fake_loss
        # 清空上一步的残余更新参数值
        # optimizerD.zero_grad()
        # # 误差反向传播, 计算参数更新值
        D_loss.backward(retain_graph=True)
        # # 将参数更新值施加到 net 的 parameters 上
        # Train Generator
        fake_img = netG(cropImg3D)
        G_loss1 = criterion(fake_img,sourceImg2D)
        G_loss_vgg = criterion(lossNetwork(torch.cat([fake_img, fake_img, fake_img], 1)),
                               lossNetwork(torch.cat([sourceImg2D, sourceImg2D, sourceImg2D], 1)))
        # 再把fake_img送入D，让D判别真假
        G_outputs = netD(cropImg3D,fake_img)
        fake_outputs = netD(cropImg3D.detach(), fake_img.detach())
        G_loss2 = torch.mean(1 - fake_outputs.mean())
        G_loss = G_loss1 + 0.02*G_loss_vgg+0.005*G_loss2

        optimizerG.zero_grad()
        G_loss.backward()
        optimizerG.step()
#
        all_D_loss = all_D_loss+D_loss.item()
        all_G_loss += G_loss.item()

        # 数据可视化
    schedulerG.step()
    schedulerD.step()
    print('Epoch {}, g_loss: {:.6f}, d_loss: {:.6f}'.format
          (epoch,  all_G_loss / (i + 1), all_D_loss / (i + 1)
               ))
        # print('Epoch {}, d_loss: {:.6f}, g_loss: {:.6f} '
        #       'D real: {:.6f}, D fake: {:.6f}'.format
        #       (epoch, all_D_loss / (i + 1), all_G_loss / (i + 1),
        #        torch.mean(real_outputs), torch.mean(fake_outputs)))

    # 将文件输出到目录中

    if (epoch%10==0):
        torch.save(netG.module.state_dict(), 'model/netG_epoch2_%d_car_%d.pth' % (30, epoch)) #训练参数保存路径
        torch.save(netD.module.state_dict(), 'model/netD_epoch2_%d_car_%d.pth' % (30, epoch))
        torch.save(netG.state_dict(), 'model/netG_epoch0000.pth' )
