# Paddle MUSA

[English](./README_en.md) | 简体中文

## 项目简介

本仓库是摩尔线程公司对 PaddlePaddle 深度学习框架进行深度适配的代码仓库，支持在摩尔线程 GPU S5000 上进行模型训练和推理。通过自定义设备接口（Custom Device），实现了 PaddlePaddle 框架与摩尔线程 GPU 硬件的无缝对接。

## 使用方式

git submodule sync
git submodule update --remote --init --recursive
```

### 编译和安装

构建脚本位于 `backends/musa/tools/build.sh`，支持以下选项：

```bash
bash tools/build.sh [选项]
```

**可用选项：**

- `-a` / `--all`: 编译 PaddlePaddle 和 paddle_musa 并安装
- `-p` / `--paddle`: 仅编译 PaddlePaddle 并安装
- `-m` / `--paddle_musa`: 仅编译 paddle_musa 并安装（依赖 PaddlePaddle 编译完成）
- `-t` / `--test`: 运行所有单元测试
- `-s` / `--single_test`: 运行单个单元测试
- `-c` / `--clean`: 清理 paddle_musa 构建文件
- `-h` / `--help`: 显示使用帮助

**编译示例：**

```bash
cd backends/musa/
# 完整编译（推荐首次使用）
bash tools/build.sh -a

# 仅编译 PaddlePaddle
bash tools/build.sh -p

# 仅编译 paddle_musa（需先完成 PaddlePaddle 编译）
bash tools/build.sh -m

# 运行所有单元测试
bash tools/build.sh -t
```

### 验证安装

```bash
# 列出可用的硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"

# 预期输出
['musa']

## 使用 PaddleInference

详见 [backends/musa/README.md](./backends/musa/README.md) 获取推理部署相关信息。

## 技术支持

如遇问题，请参考项目文档或联系摩尔线程技术支持团队。
