# 第三方组件

本发布包由启动器、单独运行的 Whisper/FFmpeg 程序及相关运行库组成。各组件保留各自的版权和许可证；首次下载的 Python 包与模型文件不包含在轻量 ZIP 中。

| 组件 | 用途 | 项目或来源 | 许可资料 |
| --- | --- | --- | --- |
| Whisper api Launcher | 启动器与服务适配代码 | https://github.com/LF12306/Whisper-api-Launcher | 根目录 `LICENSE` |
| whisper.cpp / ggml | Whisper CPU、CUDA 本地服务 | https://github.com/ggml-org/whisper.cpp | `licenses/whisper.cpp-MIT.txt` |
| OpenBLAS | Whisper CPU 运行库 | https://github.com/OpenMathLib/OpenBLAS | `licenses/OpenBLAS-BSD.txt` |
| SDL 2 | 原 Whisper 分发包中的运行库 | https://github.com/libsdl-org/SDL/tree/SDL2 | `licenses/SDL2-zlib.txt` |
| NVIDIA CUDA / cuBLAS / NVRTC | Whisper CUDA 运行库 | https://developer.nvidia.com/cuda-toolkit | NVIDIA 对应组件许可及 CUDA EULA：https://docs.nvidia.com/cuda/eula/ |
| FFmpeg | 音视频解码与转换，作为独立程序调用 | https://ffmpeg.org/ | `licenses/FFmpeg-GPLv3.txt`、`licenses/FFmpeg-build.txt` |
| uv | 独立运行环境安装工具 | https://github.com/astral-sh/uv/tree/0.13.7 | `licenses/uv-MIT.txt` |
| CPython / Tcl / Tk | 打包启动器所用的 Python 与界面运行时 | https://www.python.org/ | `licenses/Python-runtime.txt` |

随包 FFmpeg 构建版本为 `N-126905-gb87602a63a-20260927`，启用了 GPL 与 version3。其版本及完整构建参数记录在 `licenses/FFmpeg-build.txt`。对应 FFmpeg 源码提交为：https://github.com/FFmpeg/FFmpeg/tree/b87602a63a52b0f7f968fd314f02be14acde16b7 。

首次准备模型会从上游安装 PyTorch、qwen-asr 及依赖，并下载 Qwen 模型权重。相关上游项目：

- https://github.com/QwenLM/Qwen3-ASR
- https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
- https://pytorch.org/

这些运行环境和模型文件由首次安装时的上游分发提供，遵循其各自许可。
