# voice-ptt

Local hold-to-talk dictation for Linux (X11). Hold a key (or foot pedal), speak, release, and the text is typed into the focused window.

Uses faster-whisper (CUDA, with CPU fallback), `parecord` and `xdotool`. Nothing leaves the machine.

## Setup

    sudo apt install xdotool
    uv venv --python 3.13 .venv
    uv pip install --python .venv/bin/python faster-whisper pynput numpy nvidia-cublas-cu12 nvidia-cudnn-cu12
    ./run.sh

The model downloads on first run. For autostart, copy a `.desktop` entry that runs `run.sh` into `~/.config/autostart/`.

## Config

`~/.config/voice-ptt/config.json` is created on first run.

- `hotkey`: pynput key name (`ctrl_r`, `f13`, `pause`, ...). A foot pedal that sends the same key works unchanged.
- `model`, `language`, `device`, `compute_type`: Whisper settings (use `cpu` and `int8` without a GPU).
- `mic`: `pactl list short sources` name, or null for the system default.
