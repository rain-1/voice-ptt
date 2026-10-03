#!/usr/bin/env python3
"""Generate the feedback sound themes in sounds/<theme>/{start,stop}.wav.

    python make_sounds.py          regenerate every theme
    python make_sounds.py --play   regenerate, then play each theme (start, then stop) in turn

start = listening begins, stop = listening ends. Tweak a theme below and re-run. The active
theme is "sound_theme" in the config, or use "Next sound theme" in the tray menu.
"""
import os
import subprocess
import sys
import time
import wave

import numpy as np

RATE = 44100
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
VOLUME = 0.28  # peak level, 0..1; these fire on every phrase so keep them modest
rng = np.random.default_rng(1)


def t_axis(dur):
    return np.arange(int(RATE * dur)) / RATE


def seq(*parts, gap=0.0):
    pad = np.zeros(int(RATE * gap))
    out = []
    for p in parts:
        out += [p, pad]
    return np.concatenate(out)


def mix(a, b, offset=0.0):
    """Overlay b on a, starting offset seconds in."""
    start = int(RATE * offset)
    out = np.zeros(max(len(a), start + len(b)))
    out[: len(a)] += a
    out[start : start + len(b)] += b
    return out


# ---- building blocks -------------------------------------------------------------------------
def blip(freq, dur, harm2=0.25):
    t = t_axis(dur)
    w = np.sin(2 * np.pi * freq * t) + harm2 * np.sin(4 * np.pi * freq * t)
    return w * np.minimum(t / 0.004, 1) * np.exp(-t * 18 / dur)


def square(freq, dur, duty=0.5):
    t = t_axis(dur)
    w = np.where((t * freq) % 1 < duty, 1.0, -1.0)
    return w * np.minimum(t / 0.002, 1) * np.minimum((dur - t) / 0.004, 1)


def glide(f0, f1, dur, decay=10):
    t = t_axis(dur)
    f = f0 + (f1 - f0) * (t / dur)
    phase = 2 * np.pi * np.cumsum(f) / RATE
    return np.sin(phase) * np.minimum(t / 0.005, 1) * np.exp(-t * decay / dur)


def pluck(freq, dur=0.3, partials=((1, 1.0), (4, 0.35), (10, 0.1)), decay=9):
    """Marimba-ish: partials at 1x, 4x, 10x decaying fast."""
    t = t_axis(dur)
    w = sum(a * np.sin(2 * np.pi * freq * m * t) * np.exp(-t * decay * (1 + 0.3 * i)) for i, (m, a) in enumerate(partials))
    return w * np.minimum(t / 0.002, 1)


def click(center, dur=0.03, width=0.6):
    """Band-limited noise burst, like a key switch."""
    n = int(RATE * dur)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / RATE)
    spec *= np.exp(-(((f - center) / (center * width)) ** 2))
    t = t_axis(dur)
    return np.fft.irfft(spec, n) * np.exp(-t / 0.004)


def bell(freq, dur=0.4, index=2.0, ratio=1.41, decay=7):
    t = t_axis(dur)
    mod = index * np.exp(-t * 8) * np.sin(2 * np.pi * freq * ratio * t)
    return np.sin(2 * np.pi * freq * t + mod) * np.minimum(t / 0.002, 1) * np.exp(-t * decay)


def sweep(f0, f1, dur, trem=35):
    t = t_axis(dur)
    s = glide(f0, f1, dur, decay=3) * (1 + 0.4 * np.sin(2 * np.pi * trem * t))
    return mix(s, 0.4 * s, 0.07)  # one echo


# ---- themes: name -> (start, stop) -----------------------------------------------------------
THEMES = {
    # gentle rising / falling sine blips
    "soft": (seq(blip(660, 0.07), blip(990, 0.10)), seq(blip(880, 0.07), blip(587, 0.12))),
    # 8-bit arpeggios
    "retro": (
        seq(square(523, 0.045), square(659, 0.045), square(784, 0.07)),
        seq(square(784, 0.045), square(659, 0.045), square(523, 0.08)),
    ),
    # water-drop bloops: pitch glides up on start, down on stop
    "droplet": (glide(380, 1250, 0.10, decay=6), glide(1050, 330, 0.12, decay=7)),
    # wooden marimba notes: up a fifth to start, a low tock to stop
    "marimba": (mix(pluck(523, 0.25), pluck(784, 0.25), 0.07), pluck(392, 0.3)),
    # mechanical key switch: bright click to start, duller click to stop; the quietest and shortest
    "click": (click(3200, 0.025) * 1.5, click(1400, 0.035, 0.8) * 1.5),
    # bell tones, high to start, lower to stop
    "bell": (bell(1319, 0.35), bell(880, 0.45, decay=6)),
    # sci-fi sweeps with tremolo and an echo
    "scifi": (sweep(300, 1100, 0.16), sweep(1000, 260, 0.2)),
}


def write(theme, event, audio):
    audio = audio / np.abs(audio).max() * VOLUME
    d = os.path.join(OUT, theme)
    os.makedirs(d, exist_ok=True)
    with wave.open(os.path.join(d, f"{event}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    return len(audio) / RATE


if __name__ == "__main__":
    for name, (start, stop) in THEMES.items():
        print(f"{name:8s} start {write(name, 'start', start) * 1000:4.0f} ms   stop {write(name, 'stop', stop) * 1000:4.0f} ms")
    if "--play" in sys.argv:
        for name in THEMES:
            print(f"playing: {name}", flush=True)
            for event in ("start", "stop"):
                subprocess.run(["paplay", os.path.join(OUT, name, f"{event}.wav")])
                time.sleep(0.35)
            time.sleep(0.6)
