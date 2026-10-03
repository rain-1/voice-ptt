# voice-ptt

Local hold-to-talk dictation for Linux (X11). Hold a key (or foot pedal), speak, release, and the text is typed into the focused window.

Uses faster-whisper (CUDA, with CPU fallback), `parecord` and `xdotool`. Nothing leaves the machine.

## Setup

    sudo apt install xdotool
    uv venv --python 3.13 .venv
    uv pip install --python .venv/bin/python faster-whisper pynput numpy pillow nvidia-cublas-cu12 nvidia-cudnn-cu12
    ./run.sh

The model downloads on first run. For autostart, copy a `.desktop` entry that runs `run.sh` into `~/.config/autostart/`.

## Tray icon

A colour dot shows the state: grey loading, green ready, red listening, amber transcribing. Blue means idle (model not loaded), dark grey with a slash means blocked. Left-click toggles listening, right-click shows status, hotkey, model, mic and the last transcription, plus Unload model now and Block. It uses the XApp status icon through the system Python (`python3-gi`, `gir1.2-xapp`, present on Cinnamon). Set `"tray": false` to disable it.

## GPU use

The model lives in a separate worker process (`whisper_worker.py`) that starts on first use and is killed after `idle_unload_seconds` of inactivity, so the GPU is free for other work. The first phrase after an unload waits for the model to load; recording starts immediately and is transcribed once it is ready.

**Block** (tray menu) unloads the model, ignores the hotkey and remembers the setting across restarts (`~/.config/voice-ptt/state.json`) until you choose Unblock. Use it during ML runs.

## Custom vocabulary

Whisper mangles technical terms, so two layers fix them. Defaults for ML, robotics and AI coding ship in the repo, and your own files take priority. Use the tray menu's *Edit vocabulary...* and *Edit corrections...*, or edit the files directly.

- **Vocabulary** (`vocab.txt`, `~/.config/voice-ptt/vocab.txt`): one term per line. The terms are fed to Whisper as a glossary prompt, which biases it toward those spellings. The prompt is capped (~480 chars), so put the hardest terms first. It is read when the model loads, so edits apply after the next unload or restart.
- **Corrections** (`corrections.txt`, `~/.config/voice-ptt/corrections.txt`): `heard => written` per line, whole words, case-insensitive. Applied live, no restart.

Set `"vocab": false` or `"corrections": false` in the config to turn either off. See `vocab.py` for details.

## Config

`~/.config/voice-ptt/config.json` is created on first run.

- `hotkey`: pynput key name (`ctrl_r`, `f13`, `pause`, ...). A foot pedal that sends the same key works unchanged.
- `hold_ms`: 250 by default. The hotkey only counts as dictation after being held this long with no other key pressed, so taps and shortcuts like Ctrl+PageUp are ignored (audio is captured from the first instant, so no speech is lost). 0 starts immediately.
- `model`, `language`, `device`, `compute_type`: Whisper settings (use `cpu` and `int8` without a GPU).
- `preload`: load the model at startup instead of on first use. `idle_unload_seconds`: 300 by default, 0 keeps it loaded.
- `mic`: `pactl list short sources` name, or null for the system default.
