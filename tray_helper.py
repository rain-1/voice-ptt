#!/usr/bin/python3
"""Tray indicator for voice-ptt. Runs on the *system* Python (needs python3-gi + gir1.2-xapp).

Usage: tray_helper.py ICON_DIR

stdin  (from voice_ptt.py), tab-separated lines:
    state<TAB>NAME<TAB>TOOLTIP   -> switch icon to ICON_DIR/NAME.png, update status line
    info<TAB>KEY<TAB>TEXT        -> update an info line (hotkey, model, mic, last)
    info<TAB>block<TAB>on|off    -> relabel the Block menu item
stdout (to voice_ptt.py):
    toggle                       -> user left-clicked the icon or chose Start/Stop listening
    toggle_block                 -> user chose Block / Unblock
    unload                       -> user chose Unload model now
    next_sound                   -> user chose Next sound theme
    set_hotkey                   -> user chose Change hotkey (next key press becomes the hotkey)
    edit_vocab / edit_corrections -> user chose to edit their vocabulary / corrections file
Exits when stdin closes.
"""
import os
import signal
import subprocess
import sys

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("XApp", "1.0")
from gi.repository import GLib, Gtk, XApp

icon_dir = sys.argv[1]
LOG = os.path.expanduser("~/.cache/voice-ptt.log")

icon = XApp.StatusIcon()
icon.set_name("voice-ptt")


def send(cmd):
    return lambda *_: print(cmd, flush=True)


def quit_all(*_):
    try:
        os.kill(os.getppid(), signal.SIGTERM)
    finally:
        Gtk.main_quit()


def open_log(*_):
    subprocess.Popen(["xdg-open", LOG], stderr=subprocess.DEVNULL)


def label_item(text):
    item = Gtk.MenuItem(label=text)
    item.set_sensitive(False)  # display-only
    return item


menu = Gtk.Menu()
status_item = label_item("Status: starting...")
info_items = {k: label_item(f"{k.capitalize()}: ...") for k in ("hotkey", "model", "mic", "sound", "last")}
toggle_item = Gtk.MenuItem(label="Start listening")
toggle_item.connect("activate", send("toggle"))
unload_item = Gtk.MenuItem(label="Unload model now")
unload_item.connect("activate", send("unload"))
block_item = Gtk.MenuItem(label="Block (free GPU, ignore hotkey)")
block_item.connect("activate", send("toggle_block"))
sound_item = Gtk.MenuItem(label="Next sound theme")
sound_item.connect("activate", send("next_sound"))
hotkey_item = Gtk.MenuItem(label="Change hotkey...")
hotkey_item.connect("activate", send("set_hotkey"))
vocab_item = Gtk.MenuItem(label="Edit vocabulary...")
vocab_item.connect("activate", send("edit_vocab"))
corr_item = Gtk.MenuItem(label="Edit corrections...")
corr_item.connect("activate", send("edit_corrections"))
log_item = Gtk.MenuItem(label="Open log")
log_item.connect("activate", open_log)
quit_item = Gtk.MenuItem(label="Quit voice-ptt")
quit_item.connect("activate", quit_all)

menu.append(status_item)
menu.append(Gtk.SeparatorMenuItem())
for item in info_items.values():
    menu.append(item)
menu.append(Gtk.SeparatorMenuItem())
menu.append(toggle_item)
menu.append(unload_item)
menu.append(block_item)
menu.append(Gtk.SeparatorMenuItem())
menu.append(hotkey_item)
menu.append(sound_item)
menu.append(vocab_item)
menu.append(corr_item)
menu.append(log_item)
menu.append(Gtk.SeparatorMenuItem())
menu.append(quit_item)
menu.show_all()
icon.set_secondary_menu(menu)  # right click
icon.connect("activate", lambda _icon, button, _time: send("toggle")() if button == 1 else None)  # left click
icon.set_visible(True)


def set_state(name, tooltip):
    path = os.path.join(icon_dir, f"{name}.png")
    if os.path.exists(path):
        icon.set_icon_name(path)
    icon.set_tooltip_text(tooltip)
    status_item.set_label(tooltip.replace("Voice: ", "Status: "))
    toggle_item.set_label("Stop listening" if name == "recording" else "Start listening")


def set_info(key, text):
    if key == "block":
        block_item.set_label("Unblock" if text == "on" else "Block (free GPU, ignore hotkey)")
        return
    item = info_items.get(key)
    if item is not None:
        item.set_label(text.replace("_", "__"))  # GTK treats _ as a mnemonic marker


def on_stdin(fd, cond):
    line = fd.readline() if cond & GLib.IO_IN else ""
    if not line:
        Gtk.main_quit()
        return False
    kind, _, rest = line.rstrip("\n").partition("\t")
    a, _, b = rest.partition("\t")
    if kind == "state":
        set_state(a, b)
    elif kind == "info":
        set_info(a, b)
    return True


GLib.io_add_watch(sys.stdin, GLib.PRIORITY_DEFAULT, GLib.IO_IN | GLib.IO_HUP | GLib.IO_ERR, on_stdin)
Gtk.main()
