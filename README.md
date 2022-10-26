# TTTGAN
Three to Two  3D point cloud to 2D  depth image
# 一、安装所需环境

1、`pip intstall - r requirements.txt`

2、支持GPU\多GPU运行

3、torch 要从官网安装GPU版本的 不要直接 `pip install torch`

# 二、运行


## 数据对齐
为了保证输入的3D图像和标签是一一对应关系，在进行训练前将3D图像和标签重新命名并重新排序。
（请注意是否适用于你自己的数据集）

## 训练模型

`python train.py --path3D 点云图像路径 --path2D 标签图像路径 --batch_size 默认值为64（根据内存自己设置） 
                 --epochs 默认值1000 --pretrained_dictG 生成器的预训练权重，默认为空 --pretrained_dictD 鉴别器的预训练权重，默认为空
`
## 测试模型
`
python test.py --pretrained_dictG 生成器的权重参数  --img 需要被预测的图像路径`


