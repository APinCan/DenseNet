import torch.nn as nn
import torch

class DenseNet(nn.Module):
    def __init__(self, L, k):
        super(DenseNet, self).__init__()

        L = L-4
        block_num = int(L / 6)

        # input _layer
        self.intput_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1, bias=True)

        # 32x32 block
        # self.block1 = self.make_blocks(in_channels=16, block_num=block_num, growth_rate=k)
        self.block1 = DenseBlockModule(in_channels=16, block_num=block_num, growth_rate=k)
        # transition1 : output size 줄이기
        self.bn1 = nn.BatchNorm2d(num_features=k)
        # compression 세타값은 0.5로 가정, 크기를 반으로 줄임
        # 들어올때 m개의 feature-map, 세타=0.5라서 output feature-map은 두배 감소
        self.conv1 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=1, bias=True)
        self.pooling1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 16x16 block
        # self.block2 = self.make_blocks(in_channels=k, block_num=block_num, growth_rate=k)
        self.block2 = DenseBlockModule(in_channels=k, block_num=block_num, growth_rate=k)
        # transition2
        self.bn2 = nn.BatchNorm2d(num_features=k)
        self.conv2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=1, bias=True)
        self.pooling2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 8x8 block
        # self.block3 = self.make_blocks(in_channels=k, block_num=block_num, growth_rate=k)
        self.block3 = DenseBlockModule(in_channels=k, block_num=block_num, growth_rate=k)

        # last layer, global average pooling
        self.last_pooling = nn.AvgPool2d(kernel_size=8)
        self.last_linear = nn.Linear(in_features=k, out_features=10)


    def forward(self, x):
        y = self.intput_conv1(x)

        y = self.block1(y)
        y = self.bn1(y)
        y = self.conv1(y)
        y = self.pooling1(y)

        y = self.block2(y)
        y = self.bn2(y)
        y = self.conv2(y)
        y = self.pooling2(y)

        y = self.block3(y)

        y = self.last_pooling(y)
        y = y.view(y.size(0), -1)  # 이게 없으면 사이즈에러
        y = self.last_linear(y)

        return y


class DenseBlockModule(nn.Module):
    def __init__(self, in_channels, block_num, growth_rate):
        super(DenseBlockModule, self).__init__()
        self.block_num = block_num
        self.total_k = in_channels

        # self.block_list = nn.ModuleList([DenseBlock(in_channels=in_channels, growth_rate=growth_rate)])
        # block_list안에 이 모듈에서 생성해야할 모든 DenseBlock을 저장
        self.block_list = nn.ModuleList([DenseBlock(in_channels=in_channels, growth_rate=growth_rate, input_block=True)])
        self.total_k = growth_rate
        for i in range(block_num-1):
            # self.total_k += growth_rate
            self.block_list.append(DenseBlock(in_channels=self.total_k, growth_rate=growth_rate))
            self.total_k += growth_rate


    def forward(self, x):
        y_list = []
        y = None

        for i in range(self.block_num-1):
            # print('input', x.shape)
            y = self.block_list[i](x)
            y_list.append(y)

            x = torch.cat(y_list, dim=1)

        return y


class DenseBlock(nn.Module):
    def __init__(self,in_channels, growth_rate, input_block=False):
        super(DenseBlock, self).__init__()

        # DenseNet-BC
        # DenseNet-B : 1x1 conv가 있는 bottleneck
        # if input_block:
        #     self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        # else:
        #     self.bn1 = nn.BatchNorm2d(num_features=growth_rate)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)

        self.relu = nn.ReLU()
        # bottleneck layer의 1x1 conv에서는 4k개의 output feature-map 생성
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate*4, kernel_size=1, bias=True)

        self.bn2 = nn.BatchNorm2d(num_features=growth_rate*4)
        self.conv2 = nn.Conv2d(in_channels=growth_rate*4, out_channels=growth_rate, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        y = self.bn1(x)
        y = self.relu(y)
        y = self.conv1(y)

        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv2(y)

        return y

