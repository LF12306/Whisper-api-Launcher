"""Keep the upstream Japanese tokenizer usable from Unicode Windows paths."""
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile


def prepare_japanese_tokenizer():
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("nagisa")
    if spec is None:
        raise RuntimeError("缺少日语分词组件，请重新准备环境。")
    source = Path(next(iter(spec.submodule_search_locations)))
    if str(source).isascii():
        return
    model = source / "data" / "nagisa_v001.model"
    fingerprint = hashlib.sha256((str(source) + str(model.stat().st_mtime_ns)).encode()).hexdigest()[:16]
    candidates = [Path(tempfile.gettempdir()), Path(os.environ.get("PUBLIC", "C:/Users/Public"))]
    target = None
    for base in candidates:
        if str(base).isascii():
            try:
                target = base / "codex-asr-tokenizer" / fingerprint
                target.mkdir(parents=True, exist_ok=True)
                break
            except OSError:
                target = None
    if target is None:
        raise RuntimeError("日语分词组件需要可写的英文缓存路径，请检查系统临时目录。")
    if not (target / "complete.txt").exists():
        shutil.copytree(source, target / "nagisa", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (target / "complete.txt").write_text("nagisa 0.2.11", encoding="ascii")
    sys.path.insert(0, str(target))
