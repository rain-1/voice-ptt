#!/usr/bin/env python3
"""Whisper worker for voice-ptt. Runs as a child process so that killing it frees the GPU.

Usage: whisper_worker.py '<json config: model, language, device, compute_type>'

Protocol (binary pipes):
  worker -> parent: one JSON line {"ready": true, "device": "cuda"|"cpu"} once the model is loaded
  parent -> worker: "<nbytes>\\n" followed by nbytes of float32 mono 16 kHz audio
  worker -> parent: one JSON line {"text": "...", "secs": 0.27} per request
Exits when its stdin closes.
"""
import ctypes
import glob
import json
import os
import sys
import time

import numpy as np

# Keep the real stdout for the protocol; anything else that prints goes to stderr (the log).
out = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
inp = sys.stdin.buffer

cfg = json.loads(sys.argv[1])

from vocab import build_prompt, is_prompt_echo  # noqa: E402

prompt = build_prompt() if cfg.get("vocab", True) else None  # read at load, so edits apply on next load
if prompt:
    print(f"vocab prompt ({len(prompt)} chars) active", file=sys.stderr, flush=True)


def preload_cuda_libs():
    """pip-installed cuBLAS/cuDNN aren't on the loader path; load them explicitly."""
    for pkg in ("cublas", "cudnn"):
        for base in sys.path:
            for lib in sorted(glob.glob(f"{base}/nvidia/{pkg}/lib/*.so*")):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


if cfg["device"] == "cuda":
    preload_cuda_libs()
from faster_whisper import WhisperModel  # noqa: E402

device = cfg["device"]
try:
    model = WhisperModel(cfg["model"], device=device, compute_type=cfg["compute_type"])
except Exception as e:
    print(f"GPU load failed ({e}); falling back to CPU int8", file=sys.stderr, flush=True)
    device = "cpu"
    model = WhisperModel(cfg["model"], device="cpu", compute_type="int8")

# Warm up (transcribe() is lazy, so consume the segments) so the first real phrase isn't slow.
segs, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language=cfg["language"])
list(segs)

print(json.dumps({"ready": True, "device": device}), file=out, flush=True)

while True:
    line = inp.readline()
    if not line:
        break
    audio = np.frombuffer(inp.read(int(line)), dtype=np.float32).copy()
    t0 = time.time()
    segments, _ = model.transcribe(
        audio,
        language=cfg["language"],
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
        initial_prompt=prompt,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    if is_prompt_echo(text, prompt):  # Whisper parroting the glossary on near-silence
        print(f"dropped prompt echo: {text!r}", file=sys.stderr, flush=True)
        text = ""
    print(json.dumps({"text": text, "secs": time.time() - t0}), file=out, flush=True)
