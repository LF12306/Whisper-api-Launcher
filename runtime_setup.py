"""Prepare a private Python runtime and CUDA backend next to the launcher."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

UV_VERSION = "0.13.7"


def run(command, root, env):
    print("执行：" + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=root, env=env, check=True,
                   stdout=sys.stdout, stderr=sys.stderr,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def setup(root, model):
    root = Path(root).resolve()
    uv = root / ".tools" / "uv.exe"
    if not uv.exists():
        print("下载本地环境管理工具…", flush=True)
        name = "uv-x86_64-pc-windows-msvc.zip"
        base = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/"
        blob = urllib.request.urlopen(base + name, timeout=120).read()
        checksum = urllib.request.urlopen(base + name + ".sha256", timeout=30).read().decode().split()[0]
        if hashlib.sha256(blob).hexdigest() != checksum:
            raise RuntimeError("环境工具下载校验失败，请重试。")
        uv.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            uv.write_bytes(archive.read("uv.exe"))
    env = os.environ.copy()
    env.update(UV_PYTHON_INSTALL_DIR=str(root / ".runtime" / "python"),
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        run([uv, "python", "install", "3.12"], root, env)
        run([uv, "venv", "--python", "3.12", root / ".venv"], root, env)
    run([uv, "pip", "install", "--python", python, "torch==2.11.0", "torchaudio==2.11.0",
         "--index-url", "https://download.pytorch.org/whl/cu128"], root, env)
    run([uv, "pip", "install", "--python", python, "-r", root / "requirements-asr.txt"], root, env)
    if not shutil.which("ffmpeg") and not (root / "bin" / "ffmpeg.exe").exists():
        raise RuntimeError("找不到 FFmpeg。请将 ffmpeg.exe 放到 bin 文件夹，再重新准备环境。")
    run([python, root / "download_models.py", "--root", root, "--model", model], root, env)
    run([python, "-c", "from windows_compat import prepare_japanese_tokenizer; prepare_japanese_tokenizer(); from qwen_asr import Qwen3ASRModel; print('模型组件检查通过')"], root, env)
    (root / ".runtime").mkdir(exist_ok=True)
    (root / ".runtime" / "backend-ready.txt").write_text("qwen-asr=0.0.6\n", encoding="utf-8")
    print("环境和模型已就绪，可以启动服务。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model", default="qwen3-1.7b", choices=["qwen3-1.7b", "qwen3-0.6b"])
    args = parser.parse_args()
    setup(args.root, args.model)
