"""Local Qwen3-ASR service compatible with Whisper/OpenAI subtitle clients."""
import argparse
from contextlib import asynccontextmanager
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

LOG = logging.getLogger("asr")
LANGUAGES = dict(zip(
    "zh en yue ar de fr es pt id it ko ru th vi ja tr hi ms nl sv da fi pl cs fil fa el hu mk ro".split(),
    "Chinese English Cantonese Arabic German French Spanish Portuguese Indonesian Italian Korean Russian Thai Vietnamese Japanese Turkish Hindi Malay Dutch Swedish Danish Finnish Polish Czech Filipino Persian Greek Hungarian Macedonian Romanian".split(),
))
ALIGN_LANGUAGES = set("Chinese English Cantonese French German Italian Japanese Korean Portuguese Russian Spanish".split())


def normalize_language(language):
    if not language or language.lower() in {"auto", "und"}:
        return None
    value = LANGUAGES.get(language.lower(), language.title())
    if value not in LANGUAGES.values():
        raise ValueError(f"不支持的语言：{language}")
    return value


def timestamp(seconds, vtt=False):
    millis = max(0, round(seconds * 1000))
    hours, rest = divmod(millis, 3600000)
    minutes, rest = divmod(rest, 60000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{'.' if vtt else ','}{millis:03d}"


def canonical_text(text):
    """Map normalized aligner letters back to their original text spans."""
    normalized, spans, index = [], [], 0
    while index < len(text):
        end = index + 1
        while end < len(text) and (unicodedata.combining(text[end]) or text[end] in "ﾞﾟ"):
            end += 1
        for char in unicodedata.normalize("NFKC", text[index:end]).casefold():
            if char == "'" or unicodedata.category(char).startswith(("L", "N")):
                normalized.append(char)
                spans.append((index, end))
        index = end
    return "".join(normalized), spans


def aligned_words(result, duration, offset=0):
    """Retain punctuation and spaces from ASR text removed by the aligner."""
    words, cursor, previous, normalized_cursor = [], 0, 0.0, 0
    normalized, spans = canonical_text(result.text)
    for item in getattr(result.time_stamps, "items", []):
        if not (math.isfinite(item.start_time) and math.isfinite(item.end_time)):
            raise RuntimeError("对齐模型返回了无效时间戳。")
        start = min(duration, max(previous, float(item.start_time), 0))
        end = min(duration, max(start, float(item.end_time)))
        token, _ = canonical_text(item.text)
        pos = normalized.find(token, normalized_cursor)
        if token and pos >= normalized_cursor:
            begin, finish = spans[pos][0], spans[pos + len(token) - 1][1]
            prefix = result.text[cursor:begin]
            # Attach punctuation before this token to the preceding word.
            if words:
                words[-1]["word"] += prefix.rstrip()
                prefix = prefix[len(prefix.rstrip()):]
            text = prefix + result.text[begin:finish]
            cursor = finish
            normalized_cursor = pos + len(token)
        else:
            raise RuntimeError("识别文本与对齐结果不一致，无法安全生成字幕。")
        if end <= start:
            if start >= duration:
                if words:
                    words[-1]["word"] += text
                continue
            end = min(duration, start + 0.02)
        words.append(dict(word=text, start=round(offset + start, 3), end=round(offset + end, 3)))
        previous = end
    if words:
        words[-1]["word"] += result.text[cursor:]
    return words


def make_segments(words, max_chars=36, max_seconds=8):
    segments, group = [], []

    def flush():
        if group:
            segments.append(dict(id=len(segments), start=group[0]["start"], end=group[-1]["end"],
                                 text="".join(w["word"] for w in group).strip()))
            group.clear()

    for word in words:
        if group and (word["start"] - group[-1]["end"] >= 0.65 or
                      word["end"] - group[0]["start"] > max_seconds or
                      sum(len(w["word"]) for w in group) + len(word["word"]) > max_chars):
            flush()
        group.append(word)
        if word["word"].rstrip().endswith(tuple("。！？!?…")):
            flush()
    flush()
    return segments


def render_response(result, response_format):
    if response_format == "json":
        return JSONResponse({"text": result["text"]})
    if response_format == "verbose_json":
        return JSONResponse(result)
    if response_format == "text":
        return Response(result["text"], media_type="text/plain; charset=utf-8")
    vtt = response_format == "vtt"
    parts = ["WEBVTT\n"] if vtt else []
    for n, seg in enumerate(result["segments"], 1):
        parts.append(f"{n}\n{timestamp(seg['start'], vtt)} --> {timestamp(seg['end'], vtt)}\n{seg['text']}\n")
    return Response("\n".join(parts), media_type="text/vtt" if vtt else "application/x-subrip")


class QwenEngine:
    def __init__(self, model_path, aligner_path, device="auto", default_language="auto", ffmpeg=None):
        self.model_path, self.aligner_path = model_path, aligner_path
        self.device, self.default_language = device, default_language
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg")
        self.lock = threading.Lock()
        self.model = None

    def load(self):
        import torch
        from windows_compat import prepare_japanese_tokenizer
        prepare_japanese_tokenizer()
        from qwen_asr import Qwen3ASRModel

        if not self.ffmpeg:
            raise RuntimeError("找不到 FFmpeg，请把 ffmpeg.exe 放入 bin 文件夹。")
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请选择 CPU 或重新准备环境。")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        LOG.info("加载模型 %s（%s）及时间戳对齐模型…", self.model_path, self.device)
        self.model = Qwen3ASRModel.from_pretrained(
            str(self.model_path), dtype=dtype, device_map=self.device,
            attn_implementation="sdpa", local_files_only=True,
            max_inference_batch_size=1, max_new_tokens=2048,
            forced_aligner=str(self.aligner_path),
            forced_aligner_kwargs=dict(dtype=dtype, device_map=self.device,
                                       attn_implementation="sdpa", local_files_only=True),
        )
        LOG.info("模型加载完成，服务已就绪。")

    def transcribe(self, source, language, prompt, need_timestamps):
        import soundfile as sf
        from qwen_asr.inference.utils import split_audio_into_chunks

        language = normalize_language(self.default_language if language is None else language)
        if need_timestamps and language and language not in ALIGN_LANGUAGES:
            raise ValueError("此语言暂不支持时间戳，请使用 json 或 text 格式。")
        with self.lock, tempfile.TemporaryDirectory(prefix="qwen-decode-") as tmp:
            wav = Path(tmp) / "audio.wav"
            proc = subprocess.run(
                [self.ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source),
                 "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if proc.returncode:
                raise ValueError("无法解码音频：" + proc.stderr.decode("utf-8", errors="replace")[-1000:])
            audio, sample_rate = sf.read(wav, dtype="float32")
            duration = len(audio) / sample_rate
            if duration < 0.05:
                raise ValueError("音频过短或为空。")
            texts, words, detected = [], [], []
            # Short silence-aware chunks bound VRAM usage and token count for long shows.
            chunks = split_audio_into_chunks(audio, sample_rate, max_chunk_sec=30.0)
            started = time.monotonic()
            for i, (chunk, offset) in enumerate(chunks, 1):
                LOG.info("转录 %s/%s，起点 %.2f 秒", i, len(chunks), offset)
                result = self.model.transcribe(audio=(chunk, sample_rate), language=language,
                                               context=prompt, return_time_stamps=need_timestamps)[0]
                texts.append(result.text)
                if result.language:
                    detected.append(result.language)
                if need_timestamps:
                    aligned = aligned_words(result, min(len(chunk) / sample_rate, duration - offset), offset)
                    if result.text.strip() and not aligned:
                        raise RuntimeError("对齐模型未返回时间戳，请检查音频语言或重试。")
                    words.extend(aligned)
            LOG.info("完成 %.2f 秒音频，耗时 %.2f 秒。", duration, time.monotonic() - started)
            return dict(task="transcribe", language=detected[0] if detected else language or "",
                        duration=round(duration, 3), text="".join(texts),
                        segments=make_segments(words), words=words)


def create_app(engine):
    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(engine.load)
        yield

    app = FastAPI(title="本地 AI 转录服务", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": "qwen3-asr", "model": str(engine.model_path), "device": engine.device}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "whisper-1", "object": "model", "owned_by": "local"},
                                           {"id": Path(engine.model_path).name, "object": "model", "owned_by": "Qwen"}]}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>AI 转录测试</title>
<style>body{max-width:820px;margin:50px auto;font:17px system-ui;padding:20px}button,input,select{font:inherit;margin:8px}pre{white-space:pre-wrap;background:#f4f5f7;padding:20px}</style>
<h1>本地 AI 转录测试</h1><p>音频在本机处理。选择短音频，测试日语识别及字幕时间戳。</p>
<form id="f"><input name="file" type="file" accept="audio/*,video/*" required><br>
<select name="language"><option value="ja">日语</option><option value="zh">中文</option><option value="en">英语</option><option value="auto">自动识别</option></select>
<select name="response_format"><option value="srt">SRT 字幕</option><option value="verbose_json">带时间戳 JSON</option><option value="text">纯文本</option></select>
<button id="b">开始转录</button></form><pre id="out">等待选择文件…</pre><a id="save" hidden>下载结果</a>
<script>let url;f.onsubmit=async e=>{e.preventDefault();b.disabled=true;out.textContent='正在转录，请稍候…';save.hidden=true;
try{const r=await fetch('/v1/audio/transcriptions',{method:'POST',body:new FormData(f)});const t=await r.text();out.textContent=t;
if(r.ok){if(url)URL.revokeObjectURL(url);url=URL.createObjectURL(new Blob([t],{type:'text/plain;charset=utf-8'}));save.href=url;save.download='transcription.'+({srt:'srt',verbose_json:'json',text:'txt'}[f.response_format.value]);save.hidden=false;}}
catch(e){out.textContent=String(e)}finally{b.disabled=false}}</script></html>"""

    @app.post("/inference")
    @app.post("/v1/audio/transcriptions")
    @app.post("/audio/transcriptions")
    async def transcribe(request: Request):
        async with request.form(max_part_size=2 * 1024 * 1024) as form:
            upload = form.get("file")
            if not hasattr(upload, "read"):
                raise HTTPException(400, "请通过 file 字段上传音频。")
            fmt = str(form.get("response_format", "json"))
            if fmt not in {"json", "verbose_json", "srt", "vtt", "text"}:
                raise HTTPException(400, "不支持的 response_format。")
            if str(form.get("translate", "false")).lower() in {"true", "1"}:
                raise HTTPException(400, "Qwen3-ASR 仅支持转录，暂不支持翻译。")
            language = form.get("language")
            prompt = str(form.get("prompt", ""))
            try:
                with tempfile.TemporaryDirectory(prefix="qwen-upload-") as tmp:
                    path = Path(tmp) / "upload.audio"
                    total = 0
                    with path.open("wb") as output:
                        while data := await upload.read(1024 * 1024):
                            total += len(data)
                            if total > 2 * 1024 ** 3:
                                raise HTTPException(413, "单次音频不能超过 2GB，请分段后重试。")
                            output.write(data)
                    result = await run_in_threadpool(engine.transcribe, path, language, prompt,
                                                     fmt in {"verbose_json", "srt", "vtt"})
                return render_response(result, fmt)
            except ValueError as error:
                raise HTTPException(400, str(error)) from error
            except HTTPException:
                raise
            except Exception as error:
                LOG.exception("转录失败")
                if "out of memory" in str(error).lower():
                    raise HTTPException(503, "显存不足。请停止其他模型服务，或选择 0.6B 模型后重试。") from error
                raise HTTPException(500, "转录失败，请查看启动器日志：" + str(error)[:500]) from error

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--aligner", required=True)
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda:0", "cpu"])
    parser.add_argument("--language", default="auto")
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import uvicorn
    engine = QwenEngine(args.model, args.aligner, args.device, args.language, args.ffmpeg)
    uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
