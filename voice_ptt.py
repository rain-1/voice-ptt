#!/usr/bin/env python3
"""Hold-to-talk dictation: hold a key (or foot pedal), speak, release -> text is typed.

Local only: faster-whisper (CUDA, falls back to CPU) + parecord + xdotool. X11.
Config: ~/.config/voice-ptt/config.json (created on first run).
"""
import ctypes
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

CONFIG_PATH = os.path.expanduser("~/.config/voice-ptt/config.json")
DEFAULTS = {
    # pynput key name: "scroll_lock", "f9", "f13", "pause", "ctrl_r", or a single char
    "hotkey": "scroll_lock",
    "model": "large-v3-turbo",
    "language": "en",  # null = auto-detect
    "device": "cuda",  # "cuda" or "cpu"
    "compute_type": "float16",  # use "int8" on cpu
    "mic": None,  # pactl source name, null = system default
    "min_seconds": 0.3,
    "sound": True,
}


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg.update(json.load(f))
    else:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULTS, f, indent=2)
    return cfg


def preload_cuda_libs():
    """pip-installed cuBLAS/cuDNN aren't on the loader path; load them explicitly."""
    for pkg in ("cublas", "cudnn"):
        for base in sys.path:
            for lib in sorted(glob.glob(f"{base}/nvidia/{pkg}/lib/*.so*")):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def beep(cfg, name):
    if not cfg["sound"]:
        return
    path = f"/usr/share/sounds/freedesktop/stereo/{name}.oga"
    if os.path.exists(path) and shutil.which("paplay"):
        subprocess.Popen(["paplay", path], stderr=subprocess.DEVNULL)


class Recorder:
    RATE = 16000

    def __init__(self, mic):
        self.mic = mic
        self.proc = None
        self.chunks = []
        self.thread = None

    def start(self):
        cmd = ["parecord", "--raw", f"--rate={self.RATE}", "--channels=1", "--format=s16le"]
        if self.mic:
            cmd.append(f"--device={self.mic}")
        self.chunks = []
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.thread = threading.Thread(target=self._pump, args=(self.proc,), daemon=True)
        self.thread.start()

    def _pump(self, proc):
        while True:
            data = proc.stdout.read(4096)
            if not data:
                break
            self.chunks.append(data)

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is None:
            return np.zeros(0, dtype=np.float32)
        proc.terminate()
        self.thread.join(timeout=2)
        raw = b"".join(self.chunks)
        raw = raw[: len(raw) // 2 * 2]
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


_pynput_kb = None


def type_text(text):
    if shutil.which("xdotool"):
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", text],
            check=False,
        )
        return
    global _pynput_kb
    from pynput import keyboard

    if _pynput_kb is None:
        _pynput_kb = keyboard.Controller()
    _pynput_kb.type(text)


def main():
    cfg = load_config()
    if not shutil.which("parecord"):
        sys.exit("missing required tool: parecord")
    if not shutil.which("xdotool"):
        print("xdotool not found; typing via pynput (sudo apt install xdotool is more reliable)", flush=True)

    from pynput import keyboard

    key_name = cfg["hotkey"]
    hotkey = getattr(keyboard.Key, key_name, None) or keyboard.KeyCode.from_char(key_name)

    if cfg["device"] == "cuda":
        preload_cuda_libs()
    from faster_whisper import WhisperModel

    print(f"loading {cfg['model']} on {cfg['device']}...", flush=True)
    try:
        model = WhisperModel(cfg["model"], device=cfg["device"], compute_type=cfg["compute_type"])
    except Exception as e:
        print(f"GPU load failed ({e}); falling back to CPU int8", flush=True)
        model = WhisperModel(cfg["model"], device="cpu", compute_type="int8")

    rec = Recorder(cfg["mic"])
    lock = threading.Lock()
    held = False

    def transcribe_and_type(audio):
        if len(audio) < cfg["min_seconds"] * Recorder.RATE:
            return
        t0 = time.time()
        segments, _ = model.transcribe(
            audio,
            language=cfg["language"],
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"[{time.time() - t0:.2f}s] {text!r}", flush=True)
        if text:
            type_text(text + " ")

    def on_press(key):
        nonlocal held
        if key != hotkey:
            return
        with lock:
            if held:  # ignore auto-repeat
                return
            held = True
        beep(cfg, "message")
        rec.start()

    def on_release(key):
        nonlocal held
        if key != hotkey:
            return
        with lock:
            if not held:
                return
            held = False
        audio = rec.stop()
        beep(cfg, "complete")
        threading.Thread(target=transcribe_and_type, args=(audio,), daemon=True).start()

    # Warm up so the first real phrase isn't slow.
    model.transcribe(np.zeros(16000, dtype=np.float32), language=cfg["language"])
    print(f"ready: hold [{key_name}] to talk", flush=True)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
