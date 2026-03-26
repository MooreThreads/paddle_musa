# PaddlePaddle MUSA 自定义设备实现

[English](./README_en.md) | 简体中文

## 项目简介

本项目是摩尔线程（Moore Threads）公司针对 PaddlePaddle 深度学习框架进行深度适配的代码仓库，提供完整的 MUSA（Moore Threads Unified System Architecture）自定义设备支持。通过本项目的实现，用户可以在摩尔线程 GPU S5000 系列硬件上进行高效的深度学习训练和推理。

### 核心特性

- **完整的硬件支持**：支持摩尔线程 S5000 GPU 系列硬件
- **训练与推理**：同时支持模型训练和推理场景
- **分布式支持**：集成 MCCL（Moore Threads Collective Communications Library）支持多卡分布式训练
- **算子优化**：针对 MUSA 架构深度优化的算子实现
- **无缝集成**：与 PaddlePaddle 框架无缝集成，用户无需修改模型代码

## 环境准备

### 系统要求

- 操作系统：Linux (Ubuntu 20.04+)
- Python 版本：3.7+ (推荐 3.10)
- CMake 版本：3.10+
- GCC 版本：8.2+ (推荐 8.2)

### 依赖安装

```bash
# 1. 拉取 PaddlePaddle CPU 开发 Docker 镜像
# Dockerfile 位于 tools/dockerfile 目录
docker pull registry.baidubce.com/device/paddle-cpu:ubuntu20-x86_64-gcc84-py310

# 或对于 ARM 架构
docker pull registry.baidubce.com/device/paddle-cpu:ubuntu20-aarch64-gcc84-py310

# 2. 克隆源码（包含 paddle_musa 子模块）
git clone --recursive https://sh-code.mthreads.com/ai/paddle_musa.git
cd paddle_musa/backends/musa

# 3. 更新子模块
git submodule sync
git submodule update --remote --init --recursive
```

## 编译与安装

### 构建选项说明

本项目提供灵活的构建选项，可通过 `tools/build.sh` 脚本进行编译：

```bash
bash tools/build.sh [选项]
```

#### 选项详解

| 选项 | 说明 |
|------|------|
| `-a` 或 `--all` | **完整构建**：编译 PaddlePaddle 和 PaddleMUSA 并安装。这是最常用的选项，适合首次编译或完整构建。 |
| `-p` 或 `--paddle` | **仅构建 PaddlePaddle**：只编译 PaddlePaddle 框架并安装。适合只需要更新 PaddlePaddle 的场景。 |
| `-m` 或 `--paddle_musa` | **仅构建 PaddleMUSA**：只编译 MUSA 后端插件并安装。适用于 PaddlePaddle 已编译完成，只需更新 MUSA 插件的场景。 |
| `-t` 或 `--test` | **运行所有单元测试**：编译完成后运行完整的单元测试套件，验证构建的正确性。 |
| `-s` 或 `--single_test` | **运行单个单元测试**：运行指定的单个测试用例，用于调试和验证特定功能。 |
| `-c` 或 `--clean` | **清理构建**：清理所有编译产物，恢复到干净状态。适合重新编译前使用。 |
| `-h` 或 `--help` | **显示帮助信息**：显示脚本使用说明。 |

### 推荐构建流程

#### 首次完整构建

```bash
# 完整构建 PaddlePaddle 和 PaddleMUSA
bash tools/build.sh -a
```

#### 增量构建

```bash
# 只更新 PaddlePaddle
bash tools/build.sh -p

# 只更新 MUSA 后端
bash tools/build.sh -m
```

#### 清理后重建

```bash
# 清理旧的构建产物
bash tools/build.sh -c

# 重新完整构建
bash tools/build.sh -a
```

## 验证安装

### 检查设备支持

```bash
# 列出可用的硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"

# 预期输出
['musa']
```

## 目录结构

```
paddle_musa/
├── Paddle/                    # PaddlePaddle 子模块
├── backends/
│   └── musa/                  # MUSA 后端实现
│       ├── CMakeLists.txt      # 构建配置
│       ├── kernels/           # 算子实现
│       ├── runtime/           # 运行时支持
│       ├── tests/             # 测试用例
│       └── tools/             # 构建工具
├── ci/                        # CI/CD 配置
├── docker/                    # Docker 配置
├── scripts/                   # 辅助脚本
└── README.md                  # 项目文档
```

## 常见问题

### Q: 如何检查 MUSA 设备是否正确识别？

```python
import paddle
# 查看所有自定义设备类型
print(paddle.device.get_all_custom_device_type())
# 查看当前设备
print(paddle.device.get_device())
```

### Q: 编译时出现找不到 Paddle 的错误？

确保先安装了 PaddlePaddle：
```bash
bash tools/build.sh -p  # 先编译 PaddlePaddle
bash tools/build.sh -m   # 再编译 MUSA 后端
```

### Q: 如何清理并重新编译？

```bash
bash tools/build.sh -c  # 清理
bash tools/build.sh -a  # 完整重建
```

## 技术支持

- **项目主页**: https://sh-code.mthreads.com/ai/paddle_musa
- **问题反馈**: 请通过项目 issue 跟踪系统提交问题
- **文档**: 更多技术文档请参考 `docs/` 目录

## 许可证

本项目采用 Apache License 2.0 许可证，详见 [LICENSE](../../LICENSE) 文件。

## 致谢

本项目基于 PaddlePaddle 框架开发，感谢 PaddlePaddle 团队的开源贡献。
