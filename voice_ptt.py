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
import signal
import subprocess
import sys
import threading
import time

import numpy as np

import vocab

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.expanduser("~/.config/voice-ptt/config.json")
STATE_PATH = os.path.expanduser("~/.config/voice-ptt/state.json")
LOCK_PATH = os.path.expanduser("~/.cache/voice-ptt.lock")
DEFAULTS = {
    # pynput key name: "scroll_lock", "f9", "f13", "pause", "ctrl_r", or a single char
    "hotkey": "ctrl_r",
    # The hotkey only starts dictation after being held this long with no other key pressed, so
    # taps and shortcuts like Ctrl+PageUp don't trigger it. 0 = start immediately.
    "hold_ms": 250,
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
    "vocab": True,  # bias Whisper toward the glossary in vocab.txt (see vocab.py)
    "corrections": True,  # apply 'heard => written' fixes from corrections.txt
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


def save_config_value(key, value):
    """Update one key in config.json, keeping everything else."""
    with open(CONFIG_PATH) as f:
        data = json.load(f)
    data[key] = value
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def key_to_name(key):
    """pynput key -> config name (Key.ctrl_r -> 'ctrl_r', 'a' -> 'a'), or None if unnameable."""
    name = getattr(key, "name", None)
    if name:
        return name
    char = getattr(key, "char", None)
    return char if char and char.isprintable() else None


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
        self.cfg = {k: cfg[k] for k in ("model", "language", "device", "compute_type", "vocab")}
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
        # Low latency matters: with the default buffering the first data arrives ~2s after start,
        # so the beginning of a phrase is lost and short holds capture nothing.
        cmd = ["parecord", "--raw", "--latency-msec=30", f"--rate={self.RATE}", "--channels=1", "--format=s16le"]
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

    def parse_key(name):
        return getattr(keyboard.Key, name, None) or keyboard.KeyCode.from_char(name)

    key_name = cfg["hotkey"]
    hotkey = parse_key(key_name)
    capturing = False  # next key press becomes the new hotkey
    capture_timer = None  # set while waiting to see if a generic modifier press is followed by a specific one
    capture_generic = None

    corrections = vocab.Corrections()
    rec = Recorder(cfg["mic"])
    lock = threading.Lock()  # guards recording / started_by / pending
    recording = False
    committed = False  # recording confirmed as dictation (hold elapsed); False while provisional
    started_by = None
    pending = 0
    blocked = load_blocked()
    key_down = False  # hotkey physically held
    chord = False  # another key was pressed during this hold
    hold_id = 0

    def compute_state():
        if blocked:
            return "blocked"
        if recording and committed:
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
            if len(audio) < cfg["min_seconds"] * Recorder.RATE:
                print(f"too short ({len(audio) / Recorder.RATE:.2f}s), skipped", flush=True)
            elif not blocked:
                text, secs = engine.transcribe(audio)
                if cfg["corrections"]:
                    text = corrections.apply(text)
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
        """Start capturing. A hotkey recording stays provisional (silent) until commit_hold."""
        nonlocal recording, started_by, committed
        with lock:
            if recording or blocked:
                return False
            recording = True
            started_by = source
            committed = source != "key" or cfg["hold_ms"] <= 0
        rec.start()
        if committed:
            beep(cfg, "message")
            refresh()
        return True

    def cancel_key_recording(reason):
        """Discard a hotkey recording that turned out to be a tap or a shortcut."""
        nonlocal recording, committed
        with lock:
            if not recording or started_by != "key":
                return
            recording = False
            was_committed, committed = committed, False
        rec.stop()
        print(f"hotkey ignored ({reason})", flush=True)
        if was_committed:
            refresh()

    def commit_hold(my_id):
        nonlocal committed
        with lock:
            if my_id != hold_id or not key_down or chord or not recording or started_by != "key" or committed:
                return
            committed = True
        beep(cfg, "message")
        refresh()

    def stop_recording(source=None):
        """Stop and transcribe. With a source, only stops a recording that source started."""
        nonlocal recording, committed, pending
        with lock:
            if not recording or (source and started_by != source):
                return
            recording = False
            committed = False
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
        nonlocal blocked, recording, committed
        with lock:
            blocked = not blocked
            was_recording, recording = recording, False
            committed = False
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

    def hotkey_label():
        return f"Hotkey: hold {key_name} (or left-click icon)"

    def start_capture():
        nonlocal capturing
        capturing = True
        beep(cfg, "message")
        tray.info("hotkey", "Hotkey: press the new key now (Esc cancels)")
        print("press the new hotkey (Esc cancels)...", flush=True)

        def timeout():
            nonlocal capturing
            if capturing:
                capturing = False
                tray.info("hotkey", hotkey_label())
                print("hotkey change timed out", flush=True)

        threading.Timer(10, timeout).start()

    def finish_capture(key):
        nonlocal capturing, hotkey, key_name, key_down
        capturing = False
        name = key_to_name(key)
        if key == keyboard.Key.esc:
            print("hotkey change cancelled", flush=True)
        elif name is None:
            print(f"can't use {key} as a hotkey (no name); try another key", flush=True)
        else:
            hotkey, key_name, key_down = parse_key(name), name, False
            save_config_value("hotkey", name)
            beep(cfg, "complete")
            print(f"hotkey is now [{name}]", flush=True)
        tray.info("hotkey", hotkey_label())

    GENERIC_MODIFIERS = {"ctrl", "alt", "shift", "cmd"}

    def handle_capture_press(key):
        """On X11 pynput reports a right-hand modifier as a generic press (ctrl) and then the
        specific one (ctrl_r). Wait briefly so the specific key is the one that gets saved."""
        nonlocal capture_timer, capture_generic
        name = key_to_name(key)
        if capture_timer is not None:  # second event of a modifier pair
            capture_timer.cancel()
            first, capture_timer = capture_generic, None
            specific = name and name.startswith((key_to_name(first) or "?") + "_")
            finish_capture(key if specific else first)
        elif name in GENERIC_MODIFIERS:
            capture_generic = key

            def settle():
                nonlocal capture_timer
                if capture_timer is not None:  # no specific variant followed: use the generic key
                    capture_timer = None
                    finish_capture(capture_generic)

            capture_timer = threading.Timer(0.1, settle)
            capture_timer.start()
        else:
            finish_capture(key)

    def edit(path):
        vocab.ensure_user_files()
        subprocess.Popen(["xdg-open", path], stderr=subprocess.DEVNULL)

    tray = Tray(
        cfg["tray"],
        {
            "toggle": toggle,
            "toggle_block": toggle_block,
            "unload": lambda: engine.unload(),
            "set_hotkey": start_capture,
            "edit_vocab": lambda: edit(vocab.USER_VOCAB),
            "edit_corrections": lambda: edit(vocab.USER_CORRECTIONS),
        },
    )
    tray.info("hotkey", hotkey_label())
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
        nonlocal key_down, chord, hold_id
        if capturing:
            handle_capture_press(key)
            return
        if key == hotkey:
            if key_down:  # auto-repeat
                return
            key_down, chord = True, False
            hold_id += 1
            if start_recording("key") and cfg["hold_ms"] > 0:
                threading.Timer(cfg["hold_ms"] / 1000, commit_hold, args=(hold_id,)).start()
        elif key_down:  # another key during the hold: a shortcut like Ctrl+PageUp, not dictation
            chord = True
            cancel_key_recording(f"shortcut: {key} pressed during hold")

    def on_release(key):
        nonlocal key_down
        if key != hotkey:
            return
        key_down = False
        with lock:
            was_committed = committed
        if was_committed:
            stop_recording("key")  # no-op if a click started this recording
        else:
            cancel_key_recording("tap: released before hold_ms")  # discard

    signal.signal(signal.SIGUSR1, lambda *_: start_capture())
    print(f"{'BLOCKED' if blocked else 'ready'}: hold [{key_name}] to talk (model loads on first use)", flush=True)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
