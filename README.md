# Whisper api Launcher · Qwen3 / Whisper

Windows 本地语音识别 API 启动器，可配合魔改版 Aegisub 转录和打轴。界面沿用原版样式，只增加模型选择与“准备模型”按钮。

- 项目仓库：[LF12306/Whisper-api-Launcher](https://github.com/LF12306/Whisper-api-Launcher)
- 配套 Aegisub：[liuchengwucn/Aegisub](https://github.com/liuchengwucn/Aegisub)
- 发布类型：Windows x64 轻量版，2026-11-10 构建

## 首次使用

1. 将 ZIP **完整解压**到一个可写文件夹，不要直接在压缩包里启动，也不要只复制 EXE。
2. 双击 **AI转录启动器.exe**，选择 **Qwen3-ASR 1.7B**（推荐）或 **Qwen3-ASR 0.6B**（轻量）。
3. 点击 **准备模型**，等待独立运行环境和所选模型下载、安装完成。首次需要联网并下载数 GB 文件，请预留足够的磁盘空间。
4. 点击 **启动服务**，等待“服务运行中 (OpenAI Ready)”，再在 Aegisub 中开始转录。

本包不附带模型权重、本机 Python 环境或个人配置；首次准备后会在软件目录下创建 `.venv`、`.runtime` 和模型文件。无需预先安装 Python。已附带音频转换程序 FFmpeg。

识别语言由 Aegisub 的请求指定，未指定时自动识别。启动器不显示语言或设备选项，计算设备自动检测；支持 CUDA 的显卡可使用 GPU，其他情况使用 CPU。1.7B 与对齐模型在 CPU 上运行会较慢。

## Aegisub 设置

| 项目 | 填写内容 |
| --- | --- |
| Base URL / 接口地址 | `http://127.0.0.1:8086/v1` |
| Model / 模型名称 | `whisper-1` |
| Language / 语言 | 按素材选择，例如日语 `ja` |
| API Key | 任意非空文本，例如 `local` |

`whisper-1` 是兼容客户端使用的名称，实际运行的模型由启动器中的选择决定。更改启动器端口后，应同步修改 Aegisub 的接口地址。端口被占用时，请更换端口或先停止占用它的服务。

## 模型与字幕

- **Qwen3-ASR 1.7B**：质量优先，适合优先尝试。
- **Qwen3-ASR 0.6B**：较轻量，适合资源较紧张的电脑。
- **Whisper**：保留 whisper.cpp 服务支持。需自行下载 GGML 模型，放入 `models`，再在启动器中选择对应 `.bin` 文件。下载地址：[ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main)。

Qwen 模型下载使用固定 revision，地址见 `model_catalog.json`。时间戳由 Qwen3-ForcedAligner 0.6B 生成；长音频会分块处理并保留全局时间偏移。支持 SRT、VTT、纯文本、普通 JSON 和带词级时间戳的 JSON。暂不提供翻译成英语模式或说话人分离。

服务仅监听本机 `127.0.0.1`。转录音频在本机处理；“准备模型”会访问软件包和模型下载站点。点击“打开网页测试”可在本机提交短音频测试结果。

## 文件说明

- `AI转录启动器.exe`：图形化启动器，窗口图标已内嵌。
- `bin`：Whisper CPU/CUDA 服务、相关运行库及 FFmpeg，请保留。
- `.tools/uv.exe`：首次环境安装工具，请保留。
- `asr_server.py`、`windows_compat.py`：Qwen 转录服务与中文路径兼容处理，请保留。
- `download_models.py`、`runtime_setup.py`、`requirements-asr.txt`、`model_catalog.json`：环境准备与模型下载文件，请保留。
- `gui_launcher.py`、`栞子.ico`：当前界面源码与图标，供维护使用。
- `LICENSE`、`licenses`、`THIRD_PARTY_NOTICES.md`：许可证及第三方组件说明。

运行后生成的 `launcher_config.json` 保存本机设置，不需要从其他电脑复制。已准备好的本地 Python 环境包含安装位置，不应直接作为其他电脑的通用安装环境分发；换电脑时建议使用本轻量版重新准备。

## 常见问题

- **提示未准备环境或模型**：选择所需模型，点击“准备模型”，完成后再启动。
- **下载失败**：检查网络后重新点击“准备模型”；保留已有文件以便继续下载。
- **显存不足**：先停止其他大型模型服务，或切换 0.6B 模型。不要同时运行两套大型转录服务。
- **Whisper 提示模型不存在**：本轻量版不附带 GGML 权重，需要下载后手动选择。
- **识别文字或时间戳需要调整**：歌曲、背景音乐、多人重叠说话及专有名词仍建议人工复核。

## 源码与许可

启动器代码使用 MIT License。Whisper、FFmpeg、uv 及其他组件保留各自许可证，详见 `THIRD_PARTY_NOTICES.md`。模型与首次下载的软件包按各自项目的许可证提供。

维护者可使用 Python 3.12 和 PyInstaller 重新构建启动器：

```powershell
python -m PyInstaller --noconfirm --onefile --windowed --name AI转录启动器 --icon 栞子.ico --add-data "栞子.ico;." gui_launcher.py
```
