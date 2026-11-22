# Project 1: NumPy Image Encoder Implementation

本项目使用纯 NumPy 实现了 OpenCLIP ViT-B-32 的图像编码器（Image Encoder）前向计算。
(注：由于文件大小限制，本仓库未包含 weights/ 和 assets/ 目录。请从课程原附件中获取这些数据。)

## 运行环境
- Python 3.9 或更高版本
- 需要依赖库：`numpy`, `Pillow`, `imageio`

## 安装依赖
在终端中运行
```bash
python -m pip install numpy
python -m pip install Pillow
python -m pip install imageio
```

## 运行与测试
所有命令均需在当前项目文件夹（project1-numpy-student）下运行：

1.运行全部数值测试，验证实现是否正确（预期全部输出 PASS）：
```bash
python check.py --stage all
```
2.使用自带的 10 张图片生成最终的 512 维特征向量：
```bash
python run.py --output output_tokens.npz
```
3.使用自定义图片进行测试（支持传入一张或多张图片路径）：
```bash
# 测试单张图片（使用项目自带的 image_01）
python run.py --images assets/images/image_01.jpg --output my_token_01.npz
# 测试多张图片（连续输入多个路径，空格隔开）
python run.py --images assets/images/image_01.jpg assets/images/image_02.jpg --output my_tokens.npz
```