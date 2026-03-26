# PaddlePaddle MUSA Version

English | [简体中文](./README.md)

## Project Overview

This repository is a deep adaptation of the PaddlePaddle deep learning framework by Moore Threads, enabling model training and inference on Moore Threads GPU S5000. Through the Custom Device interface, it achieves seamless integration between the PaddlePaddle framework and Moore Threads GPU hardware.

## Usage

### Compilation and Installation

The build script is located at `backends/musa/tools/build.sh` and supports the following options:

```bash
bash tools/build.sh [OPTIONS]
```

**Available Options:**

- `-a` / `--all`: Build both PaddlePaddle and paddle_musa and install
- `-p` / `--paddle`: Build PaddlePaddle only and install
- `-m` / `--paddle_musa`: Build paddle_musa only and install (requires PaddlePaddle build to be completed)
- `-t` / `--test`: Run all unit tests
- `-s` / `--single_test`: Run a single unit test
- `-c` / `--clean`: Clean paddle_musa build files
- `-h` / `--help`: Display usage help

**Build Examples:**

```bash
cd backends/musa/
# Full build (recommended for first-time use)
bash tools/build.sh -a

# Build PaddlePaddle only
bash tools/build.sh -p

# Build paddle_musa only (requires PaddlePaddle build to be completed first)
bash tools/build.sh -m

# Run all unit tests
bash tools/build.sh -t
```

### Verify Installation

```bash
# List available hardware backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"

# Expected output
['musa']

# Run a simple model test
python ../tests/test_MNIST_model.py

# Expected similar output
Epoch 0 step 0, Loss = [2.2956038], Accuracy = 0.15625
Epoch 0 step 100, Loss = [2.1552896], Accuracy = 0.3125
Epoch 0 step 200, Loss = [2.1177733], Accuracy = 0.4375
Epoch 0 step 300, Loss = [2.0089214], Accuracy = 0.53125
Epoch 0 step 400, Loss = [2.0845466], Accuracy = 0.421875
Epoch 0 step 500, Loss = [2.0473], Accuracy = 0.453125
Epoch 0 step 600, Loss = [1.8561764], Accuracy = 0.71875
Epoch 0 step 700, Loss = [1.9915285], Accuracy = 0.53125
Epoch 0 step 800, Loss = [1.8925955], Accuracy = 0.640625
Epoch 0 step 900, Loss = [1.8199624], Accuracy = 0.734375
```

## Using PaddleInference

See [backends/musa/README.md](./backends/musa/README.md) for inference deployment details.

## Technical Support

If you encounter any issues, please refer to the project documentation or contact Moore Threads technical support team.
