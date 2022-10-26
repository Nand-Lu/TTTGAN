import torch.nn as nn
import torch

class Conv3(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3,stride=2,padding=1):
        super(Conv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,padding=padding),
            nn.BatchNorm3d(out_ch),  # 添加了BN层
            nn.LeakyReLU()
        )

    def forward(self, input):
        return self.conv(input)

class Dconv2(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3,padding=1,output_padding=1,stride =2):
        super(Dconv2, self).__init__()
        self.convtrans = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size,
                               stride=stride,padding=padding,output_padding=output_padding),
            nn.BatchNorm2d(out_ch),  # 添加了BN层
            nn.ReLU()
        )

    def forward(self, input):
        return self.convtrans(input)

class Conv2(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2,stride=2,padding=0):
        super(Conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,padding=padding),
            nn.BatchNorm2d(out_ch),  # 添加了BN层
            nn.LeakyReLU()
        )

    def forward(self, input):
        return self.conv(input)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential( #input  [1, 64, 32, 96]
            Conv3(1,16),        # [16, 32, 16, 48]
            Conv3(16, 64),      # [64, 16, 8, 24]
            Conv3(64, 256),     # [256, 8, 4, 12]
            Conv3(256, 512),    # [512, 4, 2, 6]
            Conv3(512, 1024),   # [1024, 2, 1, 3]
            # Conv3(1024, 2048,kernel_size=(2,1,3),stride=1),  # [2048, 1, 1, 1]
            nn.Conv3d(1024, 2048, kernel_size=(2, 1, 3), stride=(1, 1, 1)),
            nn.Tanh()

        )

        self.decoder = nn.Sequential(
            Dconv2(2048, 1024, kernel_size=(2, 1), stride=2,output_padding=(0,0),padding=0),
            Dconv2(1024, 512,  stride=2),
            Dconv2(512, 256, stride=2, ),
            Dconv2(256, 128,  stride=2),
            Dconv2(128, 64,  stride=2),
            Dconv2(64, 32,  stride=2),
        )

        self.decoder_ = nn.Sequential(
            Dconv2(40, 16 ),

            Dconv2( 16,1),
            nn.Conv2d(1,1,1,1)
        )


    def forward(self, input):

        x = self.encoder(input)
        x = torch.squeeze(x, 4)
        x = self.decoder(x)

        skip, _ = torch.topk(input, 8, dim=4, largest=True)
        skip = skip.transpose(4,1)
        skip = skip.squeeze(-1)
        x = torch.cat([skip, x], dim=1)

        x = self.decoder_(x)
        # print("x",x.size())
        x = x.transpose(3,2)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ThreeEncoder = nn.Sequential( #input  [1, 64, 32, 96]
            Conv3(1,16),        # [16, 32, 16, 48]
            Conv3(16, 64),      # [64, 16, 8, 24]
            Conv3(64, 256),     # [256, 8, 4, 12]
            Conv3(256, 512),    # [512, 4, 2, 6]
            Conv3(512, 1024),   # [1024, 2, 1, 3]
            nn.Conv3d(1024, 2048, kernel_size=(2, 1, 3), stride=(1, 1, 1)),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1)
        self.relu = nn.ReLU()
        self.TowEncoder  = nn.Sequential( #input  [1, 256,128]
            Conv2(1, 4, kernel_size=2, stride=2),
            Conv2(4, 8, kernel_size=2, stride=2),
            Conv2(8, 16, kernel_size=2, stride=2),
            Conv2(16, 32, kernel_size=2, stride=2),
            Conv2(32, 64, kernel_size=2, stride=2),
            Conv2(64, 128, kernel_size=2, stride=2),
            Conv2(128, 256, kernel_size=2, stride=2),
            # Conv2(256, 512, kernel_size=1, stride=2),
            nn.Conv2d(256, 512, kernel_size=1, stride=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        nn.Sigmoid()
        )

    def forward(self, input3d, input2d):
        x = self.ThreeEncoder(input3d)
        x = torch.squeeze(x, -1)
        x = self.conv1(x)
        x = self.relu(x)
        y = self.TowEncoder(input2d)
        x = torch.cat([x, y], dim=1)
        x = x.view(x.size()[0], -1)

        x = self.fc(x)

        return x

if __name__ == '__main__':
    i = torch.randn( 1,1, 64, 32, 96)
    g = Generator()
    output = g(i)
    print(output.shape)
    # j = torch.randn( 1,1,256,128)
    j = torch.randn(1, 1, 128, 256)
    d = Discriminator()
    out = d(i,j)
    print(out.shape)