"""Multi-model launcher using the original Whisper launcher's Tk interface."""
import ctypes
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import urllib.request
import webbrowser

ROOT = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
CONFIG = ROOT / "launcher_config.json"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def get_resource_path(relative_path):
    return str(Path(getattr(sys, "_MEIPASS", ROOT)) / relative_path)


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def default_config():
    old = read_json(ROOT / "models.json", {})
    cuda = False
    try:
        ctypes.WinDLL("nvcuda.dll")
        cuda = True
    except (OSError, AttributeError):
        pass
    return dict(engine="qwen3-1.7b", port=8086, auto_start=False,
                whisper_exe=old.get("exe_path", str(ROOT / "bin" / ("cuda" if cuda else "cpu") / "whisper-server.exe")),
                whisper_model=old.get("model_path", str(ROOT / "models" / "ggml-large-v3.bin")))


def build_command(config, catalog):
    port = int(config["port"])
    if not 1 <= port <= 65535:
        raise ValueError("端口需要在 1 到 65535 之间。")
    if config["engine"] == "whisper":
        exe, model = Path(config["whisper_exe"]).resolve(), Path(config["whisper_model"]).resolve()
        if not exe.is_file() or not model.is_file():
            raise ValueError("找不到 Whisper 程序或模型，请重新选择。")
        try:
            model_arg = os.path.relpath(model, exe.parent)
        except ValueError:
            model_arg = str(model)
        if not model_arg.isascii():
            buf = ctypes.create_unicode_buffer(32768)
            if ctypes.windll.kernel32.GetShortPathNameW(str(model), buf, len(buf)):
                model_arg = buf.value
            if not model_arg.isascii():
                raise ValueError("旧版 Whisper 引擎需要英文模型参数路径，请选择启动器内 models 文件夹中的模型。")
        return [str(exe), "-m", model_arg, "--port", str(port), "--host", "127.0.0.1",
                "--inference-path", "/v1/audio/transcriptions"], exe.parent
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.exists() or not (ROOT / ".runtime" / "backend-ready.txt").exists():
        raise ValueError("请先点击“准备模型”，首次准备需要下载数 GB 文件。")
    entry = catalog[config["engine"]]
    model = ROOT / "models" / entry["directory"]
    aligner = ROOT / "models" / catalog["aligner"]["directory"]
    if not (model / "config.json").is_file() or not (aligner / "config.json").is_file():
        raise ValueError("所选模型尚未下载，请先点击“准备模型”。")
    ffmpeg = str(ROOT / "bin" / "ffmpeg.exe") if (ROOT / "bin" / "ffmpeg.exe").exists() else shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("找不到 FFmpeg，请将 ffmpeg.exe 放到 bin 文件夹。")
    command = [str(python), str(ROOT / "asr_server.py"), "--model", str(model), "--aligner", str(aligner),
               "--port", str(port), "--ffmpeg", ffmpeg]
    return command, ROOT


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Whisper 启动器（又名：这群人在唱or念什么东西）")
        self.geometry("720x680")
        icon_path = get_resource_path("栞子.ico")
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)
        self.catalog = read_json(ROOT / "model_catalog.json", {})
        if not all(key in self.catalog for key in ("qwen3-1.7b", "qwen3-0.6b", "aligner")):
            messagebox.showerror("文件缺失", "请将启动器与 model_catalog.json 等配套文件放在同一文件夹。")
            self.destroy()
            return
        self.config = default_config()
        self.config.update(read_json(CONFIG, {}))
        self.process = None
        self.mode = None
        self.events = queue.Queue()
        self.closing = False
        self.ready = False
        self.labels = {k: self.catalog[k]["label"] for k in ("qwen3-1.7b", "qwen3-0.6b")}
        self.labels["whisper"] = "Whisper（原有 whisper.cpp 引擎）"
        if self.config["engine"] not in self.labels:
            self.config["engine"] = "qwen3-1.7b"
        self.engine = tk.StringVar(value=self.labels[self.config["engine"]])
        self.port = tk.StringVar(value=str(self.config["port"]))
        self.auto_start = tk.BooleanVar(value=self.config["auto_start"])
        self.whisper_exe = tk.StringVar(value=self.config["whisper_exe"])
        self.whisper_model = tk.StringVar(value=self.config["whisper_model"])
        self.exe_path_var = tk.StringVar(value=self.whisper_exe.get())
        self.model_path_var = tk.StringVar(value=self.whisper_model.get())
        self._display_engine = None
        self.controls = []

        # Preserve the original layout, colors, fonts, title, and window size.
        config_frame = tk.LabelFrame(self, text="基础配置")
        config_frame.pack(pady=10, padx=10, fill="x")
        self.path_controls = []
        for row, label, variable, pattern in [(0, "Server程序:", self.exe_path_var, "*.exe"),
                                              (1, "模型路径:", self.model_path_var, "*.bin")]:
            tk.Label(config_frame, text=label).grid(row=row, column=0, sticky="w", padx=5)
            entry = tk.Entry(config_frame, textvariable=variable, width=60)
            entry.grid(row=row, column=1, padx=5)
            button = tk.Button(config_frame, text="...", command=lambda v=variable, p=pattern: self.browse(v, p))
            button.grid(row=row, column=2)
            self.path_controls.extend([entry, button])
        tk.Label(config_frame, text="端口 (Port):").grid(row=2, column=0, sticky="w", padx=5)
        port_entry = tk.Entry(config_frame, textvariable=self.port, width=10)
        port_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        tk.Checkbutton(config_frame, text="打开程序时自动启动服务", variable=self.auto_start).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # Only the controls needed by the new models are added below the old fields.
        tk.Label(config_frame, text="识别模型:").grid(row=4, column=0, sticky="w", padx=5)
        model_box = tk.OptionMenu(config_frame, self.engine, *self.labels.values(), command=self.engine_changed)
        model_box.configure(width=39, anchor="w")
        model_box.grid(row=4, column=1, sticky="w", padx=5, pady=2)
        self.prepare = tk.Button(config_frame, text="准备模型", command=self.prepare_backend)
        self.prepare.grid(row=4, column=2, padx=(0, 5))
        self.controls.extend([model_box, port_entry])
        self.port.trace_add("write", lambda *_: self.update_url())

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        self.start = tk.Button(btn_frame, text="启动服务", command=self.toggle_server,
                               bg="#e1f5fe", font=("微软雅黑", 14, "bold"), width=15, height=2)
        self.start.pack(side="left", padx=10)
        self.test = tk.Button(btn_frame, text="打开网页测试", command=self.open_test,
                              state="disabled", font=("微软雅黑", 10))
        self.test.pack(side="left", padx=10)
        self.status = tk.StringVar(value="状态: 已停止")
        self.status_label = tk.Label(self, textvariable=self.status, fg="#555", font=("微软雅黑", 10, "bold"))
        self.status_label.pack(pady=5)

        info_frame = tk.Frame(self, bg="#e8f5e9", borderwidth=1, relief="solid")
        info_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(info_frame, text="✅ 兼容模式已开启，请复制下方地址到打轴软件：",
                 bg="#e8f5e9", fg="#1b5e20").pack(pady=2)
        self.url = tk.StringVar()
        self.url_label = tk.Entry(info_frame, textvariable=self.url, state="readonly", font=("Consolas", 11, "bold"),
                                  fg="#2e7d32", bg="#e8f5e9", justify="center", bd=0)
        self.url_label.pack(fill="x", padx=20, pady=5)

        log_frame = tk.LabelFrame(self, text="运行日志")
        log_frame.pack(pady=5, padx=10, fill="both", expand=True)
        self.log_area = scrolledtext.ScrolledText(log_frame, state="disabled", height=10, font=("Consolas", 9))
        self.log_area.pack(fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.engine_changed()
        self.update_url()
        self.after(100, self.pump)
        if self.auto_start.get():
            self.after(600, self.toggle_server)

    def selected_engine(self):
        return next(k for k, label in self.labels.items() if label == self.engine.get())

    def engine_changed(self, _event=None):
        if self._display_engine == "whisper":
            self.whisper_exe.set(self.exe_path_var.get())
            self.whisper_model.set(self.model_path_var.get())
        key = self.selected_engine()
        if key == "whisper":
            self.exe_path_var.set(self.whisper_exe.get())
            self.model_path_var.set(self.whisper_model.get())
        else:
            self.exe_path_var.set(str(ROOT / ".venv" / "Scripts" / "python.exe"))
            self.model_path_var.set(str(ROOT / "models" / self.catalog[key]["directory"]))
        self._display_engine = key
        self.set_controls(self.process is None)

    def update_url(self):
        if hasattr(self, "url"):
            self.url.set(f"http://127.0.0.1:{self.port.get()}/v1")

    def current_config(self):
        if self.selected_engine() == "whisper":
            self.whisper_exe.set(self.exe_path_var.get())
            self.whisper_model.set(self.model_path_var.get())
        return dict(engine=self.selected_engine(), port=int(self.port.get()), auto_start=self.auto_start.get(),
                    whisper_exe=self.whisper_exe.get(), whisper_model=self.whisper_model.get())

    def save_config(self):
        data = self.current_config()
        temporary = CONFIG.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(CONFIG)

    def browse(self, variable, pattern):
        path = filedialog.askopenfilename(filetypes=[("所需文件", pattern), ("所有文件", "*.*")])
        if path:
            variable.set(path)

    def log(self, text):
        self.log_area.configure(state="normal")
        self.log_area.insert("end", text)
        if int(self.log_area.index("end-1c").split(".")[0]) > 5000:
            self.log_area.delete("1.0", "1000.0")
        self.log_area.see("end")
        self.log_area.configure(state="disabled")

    def set_controls(self, enabled):
        whisper_enabled = enabled and self.selected_engine() == "whisper"
        for widget in self.controls:
            widget.configure(state="normal" if enabled else "disabled")
        for widget in self.path_controls:
            if isinstance(widget, tk.Entry):
                widget.configure(state="normal" if whisper_enabled else "readonly" if enabled else "disabled")
            else:
                widget.configure(state="normal" if whisper_enabled else "disabled")
        self.prepare.configure(state="normal" if enabled and self.selected_engine() != "whisper" else "disabled")

    def launch(self, command, cwd, mode):
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        setup_log = None
        if mode == "setup" and getattr(sys, "frozen", False):
            setup_log = ROOT / ".runtime" / "setup-output.log"
            setup_log.parent.mkdir(exist_ok=True)
            setup_log.write_text("", encoding="utf-8")
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        self.process, self.mode, self.ready = proc, mode, False
        self.set_controls(False)
        self.test.configure(state="disabled")
        self.start.configure(text="停止准备" if mode == "setup" else "停止启动", bg="#ffe0b2")
        self.status_label.configure(fg="orange")
        self.status.set("状态: 正在准备环境及模型…" if mode == "setup" else "状态: 🚀 正在启动 (加载 OpenAI 兼容模式)...")

        def read_output():
            if setup_log:
                with setup_log.open(encoding="utf-8", errors="replace") as stream:
                    while True:
                        line = stream.readline()
                        if line:
                            self.events.put(("log", line))
                        elif proc.poll() is not None:
                            break
                        else:
                            threading.Event().wait(0.1)
            else:
                for line in iter(proc.stdout.readline, b""):
                    self.events.put(("log", line.decode("utf-8", errors="replace")))
            proc.stdout.close()
            self.events.put(("exit", proc, proc.wait()))

        threading.Thread(target=read_output, daemon=True).start()
        if mode == "server":
            port, engine = self.port.get(), self.selected_engine()

            def check_health():
                url = f"http://127.0.0.1:{port}/" + ("health" if engine != "whisper" else "")
                while proc.poll() is None:
                    try:
                        with urllib.request.urlopen(url, timeout=1) as response:
                            if response.status == 200:
                                self.events.put(("ready", proc))
                                return
                    except OSError:
                        pass
                    threading.Event().wait(1)

            threading.Thread(target=check_health, daemon=True).start()

    def prepare_backend(self):
        if self.process:
            return
        try:
            self.save_config()
            if getattr(sys, "frozen", False):
                command = [sys.executable, "--setup-backend", self.selected_engine()]
            else:
                command = [sys.executable, str(ROOT / "runtime_setup.py"), "--root", str(ROOT), "--model", self.selected_engine()]
            self.launch(command, ROOT, "setup")
        except Exception as error:
            messagebox.showerror("准备失败", str(error))

    def toggle_server(self):
        if self.process:
            self.stop_process()
            return
        try:
            config = self.current_config()
            command, cwd = build_command(config, self.catalog)
            with socket.socket() as check:
                if check.connect_ex(("127.0.0.1", config["port"])) == 0:
                    raise ValueError(f"端口 {config['port']} 已被其他服务占用。请选择另一个端口，例如 {config['port'] + 1}。")
            self.save_config()
            self.launch(command, cwd, "server")
        except Exception as error:
            messagebox.showerror("无法启动", str(error))

    def stop_process(self):
        proc = self.process
        if proc and proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
            else:
                proc.terminate()

    def pump(self):
        for _ in range(200):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "log":
                self.log(event[1])
            elif event[0] == "ready" and event[1] is self.process and self.process.poll() is None:
                self.ready = True
                self.status.set("状态: ✅ 服务运行中 (OpenAI Ready)")
                self.status_label.configure(fg="green")
                self.start.configure(text="停止服务", bg="#ffcdd2")
                self.test.configure(state="normal")
            elif event[0] == "exit" and event[1] is self.process:
                mode, code = self.mode, event[2]
                self.process, self.mode, self.ready = None, None, False
                self.set_controls(True)
                self.start.configure(text="启动服务", bg="#e1f5fe")
                self.status_label.configure(fg="red")
                self.test.configure(state="disabled")
                self.status.set("状态: 准备完成，请点击启动服务" if mode == "setup" and code == 0 else
                                f"状态: ⛔ 已停止 (代码 {code})" if code else "状态: ⛔ 已停止")
        if not self.closing:
            self.after(100, self.pump)

    def copy_url(self):
        self.clipboard_clear()
        self.clipboard_append(self.url.get())

    def open_test(self):
        webbrowser.open(f"http://127.0.0.1:{self.port.get()}/")

    def on_closing(self):
        try:
            self.save_config()
        except (OSError, ValueError):
            pass
        self.closing = True
        self.stop_process()
        self.destroy()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--setup-backend":
        if getattr(sys, "frozen", False):
            (ROOT / ".runtime").mkdir(exist_ok=True)
            sys.stdout = sys.stderr = (ROOT / ".runtime" / "setup-output.log").open("a", encoding="utf-8", buffering=1)
        from runtime_setup import setup
        try:
            setup(ROOT, sys.argv[2])
        except Exception:
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        Application().mainloop()


if __name__ == "__main__":
    main()
