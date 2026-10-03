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
    "tray": True,  # colour dot in the system tray showing state
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


STATES = {
    # state: (colour, tooltip)
    "loading": ((128, 128, 128), "Voice: loading model..."),
    "ready": ((46, 160, 67), "Voice: ready"),
    "recording": ((220, 38, 38), "Voice: listening"),
    "transcribing": ((245, 158, 11), "Voice: transcribing"),
}


class Tray:
    """System tray dot showing the current state, via tray_helper.py (XApp, system Python).

    Left click calls on_toggle(); right click shows a menu with state and info lines.
    Silently disabled if unavailable.
    """

    def __init__(self, enabled, on_toggle):
        self.proc = None
        self.on_toggle = on_toggle
        if not enabled:
            return
        try:
            from PIL import Image, ImageDraw

            icon_dir = os.path.expanduser("~/.cache/voice-ptt/icons")
            os.makedirs(icon_dir, exist_ok=True)
            for name, (colour, _) in STATES.items():
                img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                ImageDraw.Draw(img).ellipse((6, 6, 58, 58), fill=colour + (255,))
                img.save(os.path.join(icon_dir, f"{name}.png"))
            helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tray_helper.py")
            self.proc = subprocess.Popen(
                ["/usr/bin/python3", helper, icon_dir],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            threading.Thread(target=self._read_clicks, args=(self.proc,), daemon=True).start()
        except Exception as e:
            print(f"tray disabled: {e}", flush=True)
            self.proc = None

    def _read_clicks(self, proc):
        for line in proc.stdout:
            if line.strip() == "toggle":
                self.on_toggle()

    def _send(self, line):
        if self.proc is None:
            return
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except Exception:
            self.proc = None

    def set(self, state):
        self._send(f"state\t{state}\t{STATES[state][1]}")

    def info(self, key, text):
        self._send(f"info\t{key}\t{text.replace(chr(10), ' ')}")


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

    tray = Tray(cfg["tray"], lambda: toggle())  # toggle is defined below; looked up at click time
    tray.set("loading")

    if cfg["device"] == "cuda":
        preload_cuda_libs()
    from faster_whisper import WhisperModel

    print(f"loading {cfg['model']} on {cfg['device']}...", flush=True)
    try:
        model = WhisperModel(cfg["model"], device=cfg["device"], compute_type=cfg["compute_type"])
    except Exception as e:
        print(f"GPU load failed ({e}); falling back to CPU int8", flush=True)
        model = WhisperModel(cfg["model"], device="cpu", compute_type="int8")

    device = cfg["device"] if model.model.device == cfg["device"] else "cpu"
    model_info = f"Model: {cfg['model']} ({device})"
    if device != cfg["device"]:
        model_info += " - GPU unavailable"
    tray.info("hotkey", f"Hotkey: hold {key_name} (or left-click icon)")
    tray.info("model", model_info)
    tray.info("mic", f"Mic: {cfg['mic'] or 'system default'}")
    tray.info("last", "Last: (nothing yet)")

    rec = Recorder(cfg["mic"])
    lock = threading.Lock()  # guards recording / started_by / pending
    transcribe_lock = threading.Lock()  # one transcription at a time
    recording = False
    started_by = None
    pending = 0

    def transcribe_and_type(audio):
        nonlocal pending
        try:
            if len(audio) < cfg["min_seconds"] * Recorder.RATE:
                return
            with transcribe_lock:
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
                shown = text if len(text) <= 60 else text[:57] + "..."
                tray.info("last", f"Last: {shown}")
        finally:
            with lock:
                pending -= 1
                state = "recording" if recording else "transcribing" if pending else "ready"
            tray.set(state)

    def start_recording(source):
        nonlocal recording, started_by
        with lock:
            if recording:
                return
            recording = True
            started_by = source
        tray.set("recording")
        beep(cfg, "message")
        rec.start()

    def stop_recording(source=None):
        """Stop and transcribe. With a source, only stops a recording that source started."""
        nonlocal recording, pending
        with lock:
            if not recording or (source and started_by != source):
                return
            recording = False
            pending += 1
        audio = rec.stop()
        tray.set("transcribing")
        beep(cfg, "complete")
        threading.Thread(target=transcribe_and_type, args=(audio,), daemon=True).start()

    def toggle():
        if recording:
            stop_recording()
        else:
            start_recording("click")

    def on_press(key):
        if key == hotkey:  # auto-repeat presses are ignored by start_recording
            start_recording("key")

    def on_release(key):
        if key == hotkey:
            stop_recording("key")

    # Warm up so the first real phrase isn't slow.
    model.transcribe(np.zeros(16000, dtype=np.float32), language=cfg["language"])
    tray.set("ready")
    print(f"ready: hold [{key_name}] to talk", flush=True)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
