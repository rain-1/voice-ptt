#!/usr/bin/env python3
"""Hold-to-talk dictation: hold a key (or foot pedal), speak, release -> text is typed.

Local only: faster-whisper (CUDA, falls back to CPU) + parecord + xdotool. X11.
The model runs in a worker process that is started on first use and stopped after an idle
period (or by "Block" in the tray menu), so the GPU is free for other work.
Config: ~/.config/voice-ptt/config.json (created on first run).
"""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.expanduser("~/.config/voice-ptt/config.json")
STATE_PATH = os.path.expanduser("~/.config/voice-ptt/state.json")
LOCK_PATH = os.path.expanduser("~/.cache/voice-ptt.lock")
DEFAULTS = {
    # pynput key name: "scroll_lock", "f9", "f13", "pause", "ctrl_r", or a single char
    "hotkey": "ctrl_r",
    "model": "large-v3-turbo",
    "language": "en",  # null = auto-detect
    "device": "cuda",  # "cuda" or "cpu"
    "compute_type": "float16",  # use "int8" on cpu
    "mic": None,  # pactl source name, null = system default
    "min_seconds": 0.3,
    "sound": True,
    "tray": True,  # colour dot in the system tray showing state
    "preload": False,  # load the model at startup instead of on first use
    "idle_unload_seconds": 300,  # unload the model (free the GPU) after this long idle; 0 = never
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


def load_blocked():
    try:
        with open(STATE_PATH) as f:
            return bool(json.load(f).get("blocked"))
    except (OSError, ValueError):
        return False


def save_blocked(blocked):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({"blocked": blocked}, f)


def beep(cfg, name):
    if not cfg["sound"]:
        return
    path = f"/usr/share/sounds/freedesktop/stereo/{name}.oga"
    if os.path.exists(path) and shutil.which("paplay"):
        subprocess.Popen(["paplay", path], stderr=subprocess.DEVNULL)


STATES = {
    # state: (colour, tooltip)
    "idle": ((59, 130, 246), "Voice: idle (model unloaded)"),
    "loading": ((128, 128, 128), "Voice: loading model..."),
    "ready": ((46, 160, 67), "Voice: ready"),
    "recording": ((220, 38, 38), "Voice: listening"),
    "transcribing": ((245, 158, 11), "Voice: transcribing"),
    "blocked": ((90, 90, 90), "Voice: blocked (GPU free)"),
}


class Tray:
    """System tray dot showing the current state, via tray_helper.py (XApp, system Python).

    Left click calls on_toggle(); the menu offers block and unload. Silently disabled if
    unavailable.
    """

    def __init__(self, enabled, handlers):
        self.proc = None
        self.handlers = handlers  # helper stdout line -> callable
        if not enabled:
            return
        try:
            from PIL import Image, ImageDraw

            icon_dir = os.path.expanduser("~/.cache/voice-ptt/icons")
            os.makedirs(icon_dir, exist_ok=True)
            for name, (colour, _) in STATES.items():
                img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                d = ImageDraw.Draw(img)
                d.ellipse((6, 6, 58, 58), fill=colour + (255,))
                if name == "blocked":
                    d.line((18, 46, 46, 18), fill=(255, 255, 255, 255), width=7)
                img.save(os.path.join(icon_dir, f"{name}.png"))
            self.proc = subprocess.Popen(
                ["/usr/bin/python3", os.path.join(HERE, "tray_helper.py"), icon_dir],
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
            handler = self.handlers.get(line.strip())
            if handler:
                try:
                    handler()
                except Exception as e:
                    print(f"tray action {line.strip()!r} failed: {e}", flush=True)

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


class Engine:
    """Owns the whisper worker process: started on demand, killed to free the GPU."""

    def __init__(self, cfg, on_change):
        self.cfg = {k: cfg[k] for k in ("model", "language", "device", "compute_type")}
        self.on_change = on_change
        self.lock = threading.Lock()  # serialises requests and start/stop
        self.proc = None
        self.device = None
        self.last_used = time.time()

    @property
    def loaded(self):
        return self.proc is not None and self.proc.poll() is None

    def _ensure_started(self):
        if self.loaded:
            return
        print(f"loading {self.cfg['model']} on {self.cfg['device']}...", flush=True)
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "whisper_worker.py"), json.dumps(self.cfg)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        line = self.proc.stdout.readline()
        if not line:
            self.proc = None
            raise RuntimeError("whisper worker exited during startup (see log)")
        self.device = json.loads(line)["device"]
        print(f"model ready on {self.device}", flush=True)
        self.on_change()

    def warm(self):
        with self.lock:
            self._ensure_started()
            self.last_used = time.time()

    def transcribe(self, audio):
        data = audio.astype(np.float32).tobytes()
        with self.lock:
            self._ensure_started()
            self.proc.stdin.write(f"{len(data)}\n".encode() + data)
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            self.last_used = time.time()
            if not line:
                self.proc = None
                raise RuntimeError("whisper worker died (see log)")
        reply = json.loads(line)
        return reply["text"], reply["secs"]

    def unload(self):
        with self.lock:
            proc, self.proc = self.proc, None
            if proc is None:
                return
            proc.stdin.close()  # worker exits on EOF
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        print("model unloaded", flush=True)
        self.on_change()


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

    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    lock_file = open(LOCK_PATH, "w")  # held for the life of the process
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("voice-ptt is already running")

    from pynput import keyboard

    key_name = cfg["hotkey"]
    hotkey = getattr(keyboard.Key, key_name, None) or keyboard.KeyCode.from_char(key_name)

    rec = Recorder(cfg["mic"])
    lock = threading.Lock()  # guards recording / started_by / pending
    recording = False
    started_by = None
    pending = 0
    blocked = load_blocked()

    def compute_state():
        if blocked:
            return "blocked"
        if recording:
            return "recording"
        if pending:
            return "transcribing" if engine.loaded else "loading"
        return "ready" if engine.loaded else "idle"

    def refresh():
        tray.set(compute_state())
        tray.info("block", "on" if blocked else "off")
        if engine.loaded:
            note = f"{engine.device}, loaded"
            if engine.device != cfg["device"]:
                note += " - GPU unavailable"
        else:
            note = "not loaded, GPU free"
        tray.info("model", f"Model: {cfg['model']} ({note})")

    engine = Engine(cfg, on_change=lambda: refresh())

    def transcribe_and_type(audio):
        nonlocal pending
        try:
            if len(audio) >= cfg["min_seconds"] * Recorder.RATE and not blocked:
                text, secs = engine.transcribe(audio)
                print(f"[{secs:.2f}s] {text!r}", flush=True)
                if text:
                    type_text(text + " ")
                    shown = text if len(text) <= 60 else text[:57] + "..."
                    tray.info("last", f"Last: {shown}")
        except Exception as e:
            print(f"transcription failed: {e}", flush=True)
        finally:
            with lock:
                pending -= 1
            refresh()

    def start_recording(source):
        nonlocal recording, started_by
        with lock:
            if recording or blocked:
                return
            recording = True
            started_by = source
        beep(cfg, "message")
        rec.start()
        refresh()

    def stop_recording(source=None):
        """Stop and transcribe. With a source, only stops a recording that source started."""
        nonlocal recording, pending
        with lock:
            if not recording or (source and started_by != source):
                return
            recording = False
            pending += 1
        audio = rec.stop()
        beep(cfg, "complete")
        refresh()
        threading.Thread(target=transcribe_and_type, args=(audio,), daemon=True).start()

    def toggle():
        if recording:
            stop_recording()
        else:
            start_recording("click")

    def toggle_block():
        nonlocal blocked, recording
        with lock:
            blocked = not blocked
            was_recording, recording = recording, False
        save_blocked(blocked)
        if blocked:
            if was_recording:
                rec.stop()  # discard the audio
            refresh()
            engine.unload()
            print("blocked: model unloaded, hotkey ignored", flush=True)
        else:
            refresh()
            print("unblocked", flush=True)
            if cfg["preload"]:
                threading.Thread(target=engine.warm, daemon=True).start()

    tray = Tray(
        cfg["tray"],
        {"toggle": toggle, "toggle_block": toggle_block, "unload": lambda: engine.unload()},
    )
    tray.info("hotkey", f"Hotkey: hold {key_name} (or left-click icon)")
    tray.info("mic", f"Mic: {cfg['mic'] or 'system default'}")
    tray.info("last", "Last: (nothing yet)")
    refresh()

    def idle_watcher():
        while True:
            time.sleep(5)
            idle = time.time() - engine.last_used
            if engine.loaded and not recording and not pending and idle > cfg["idle_unload_seconds"]:
                print(f"idle for {idle:.0f}s, unloading model", flush=True)
                engine.unload()

    if cfg["idle_unload_seconds"] > 0:
        threading.Thread(target=idle_watcher, daemon=True).start()
    if cfg["preload"] and not blocked:
        threading.Thread(target=engine.warm, daemon=True).start()

    def on_press(key):
        if key == hotkey:  # auto-repeat presses are ignored by start_recording
            start_recording("key")

    def on_release(key):
        if key == hotkey:
            stop_recording("key")

    print(f"{'BLOCKED' if blocked else 'ready'}: hold [{key_name}] to talk (model loads on first use)", flush=True)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
