#!/usr/bin/env bash
# Set up voice-ptt: system packages, Python venv, autostart entry, model download.
# Usage: ./install.sh [--no-apt] [--no-model] [--uninstall]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
AUTOSTART="${XDG_CONFIG_HOME:-$HOME/.config}/autostart/voice-ptt.desktop"
APT_PKGS=(xdotool pulseaudio-utils python3-gi gir1.2-xapp-1.0)
DO_APT=1 DO_MODEL=1

for arg in "$@"; do
  case "$arg" in
    --no-apt) DO_APT=0 ;;
    --no-model) DO_MODEL=0 ;;
    --uninstall)
      pkill -f "[.]venv/bin/python voice_ptt" 2>/dev/null || true
      rm -f "$AUTOSTART"
      echo "Removed autostart entry and stopped the service. Config in ~/.config/voice-ptt and $HERE left alone."
      exit 0 ;;
    -h|--help) sed -n '2,3p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n==> %s\n' "$*"; }

[ "$(uname -s)" = Linux ] || { echo "Linux only." >&2; exit 1; }
[ "${XDG_SESSION_TYPE:-x11}" = x11 ] || echo "warning: session is ${XDG_SESSION_TYPE}; the hotkey listener and xdotool need X11."

# 1. System packages (Debian/Ubuntu). Only touches sudo if something is missing.
if [ "$DO_APT" = 1 ]; then
  say "System packages"
  if command -v dpkg >/dev/null; then
    missing=()
    for p in "${APT_PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
    if [ ${#missing[@]} -eq 0 ]; then
      echo "all present: ${APT_PKGS[*]}"
    else
      echo "installing: ${missing[*]}"
      sudo apt-get install -y "${missing[@]}" || echo "apt failed; run: sudo apt install ${missing[*]}  (then re-run, or use --no-apt)"
    fi
  else
    echo "not a dpkg system; make sure these are installed: ${APT_PKGS[*]} (names may differ)"
  fi
fi

# 2. Python venv + packages
say "Python environment"
PKGS=(faster-whisper pynput numpy pillow)
if command -v nvidia-smi >/dev/null 2>&1 || [ -e /proc/driver/nvidia/version ]; then
  PKGS+=(nvidia-cublas-cu12 nvidia-cudnn-cu12)
  echo "NVIDIA GPU detected: adding CUDA libraries"
else
  echo "no NVIDIA GPU detected: set \"device\": \"cpu\" and \"compute_type\": \"int8\" in ~/.config/voice-ptt/config.json"
fi
if command -v uv >/dev/null; then
  [ -x "$HERE/.venv/bin/python" ] || uv venv --python 3.13 "$HERE/.venv" 2>/dev/null || uv venv "$HERE/.venv"
  uv pip install --quiet --python "$HERE/.venv/bin/python" "${PKGS[@]}"
else
  [ -x "$HERE/.venv/bin/python" ] || python3 -m venv "$HERE/.venv"
  "$HERE/.venv/bin/pip" install --quiet "${PKGS[@]}"
fi
echo "venv ready: $HERE/.venv"

# 3. Autostart entry (login). Nothing is loaded onto the GPU until first use.
say "Autostart"
mkdir -p "$(dirname "$AUTOSTART")" "$HOME/.cache"
cat > "$AUTOSTART" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Voice PTT
Comment=Hold-to-talk local dictation
Exec=sh -c "$HERE/run.sh >> $HOME/.cache/voice-ptt.log 2>&1"
X-GNOME-Autostart-enabled=true
DESKTOP
echo "wrote $AUTOSTART"

# 4. Pre-download the model so the first use isn't a long download.
if [ "$DO_MODEL" = 1 ]; then
  say "Whisper model"
  if "$HERE/.venv/bin/python" - <<PY
import sys
sys.path.insert(0, "$HERE")
import voice_ptt
from faster_whisper.utils import download_model

model = voice_ptt.load_config()["model"]
print(f"downloading {model} (cached after the first time)...")
download_model(model)
print(f"model ready: {model}")
PY
  then :; else echo "model download failed; it will be retried on first use"; fi
fi

say "Done"
if pgrep -f "[.]venv/bin/python voice_ptt" >/dev/null; then
  echo "voice-ptt is already running. Restart it to pick up changes: ./install.sh --uninstall && ./install.sh"
else
  echo "Starting now (also starts at next login)."
  (setsid nohup "$HERE/run.sh" >> "$HOME/.cache/voice-ptt.log" 2>&1 &)
fi
echo "Hold the hotkey (default Right Ctrl) and talk. Tray icon: right-click for options."
