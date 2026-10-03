"""Custom vocabulary for voice-ptt: a Whisper prompt glossary and post-hoc spelling corrections.

Two layers of files, both plain text:
  shipped:  vocab.txt, corrections.txt (in this repo)
  yours:    ~/.config/voice-ptt/vocab.txt, ~/.config/voice-ptt/corrections.txt (take priority)

vocab.txt        one term per line; '#' starts a comment. Fed to Whisper as its prompt, which
                 biases it toward these spellings. Hardest terms first: the prompt is capped.
corrections.txt  'heard => written' per line, matched case-insensitively on whole words;
                 spaces in 'heard' also match hyphens or several spaces.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
USER_DIR = os.path.expanduser("~/.config/voice-ptt")
USER_VOCAB = os.path.join(USER_DIR, "vocab.txt")
USER_CORRECTIONS = os.path.join(USER_DIR, "corrections.txt")

# Whisper only reads the last ~223 tokens of the prompt; unusual terms cost 2-3 tokens each.
PROMPT_MAX_CHARS = 480

USER_VOCAB_TEMPLATE = """# Your own vocabulary: one term per line (names, libraries, robots, projects).
# These go first in the prompt, ahead of the defaults in the repo's vocab.txt.
"""
USER_CORRECTIONS_TEMPLATE = """# Your own corrections: 'heard => written', one per line. Applied before the defaults.
# Example:
#   pie torch => PyTorch
"""


def _lines(path):
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    yield line
    except OSError:
        return


def ensure_user_files():
    """Create the user's files with a header if missing, so 'edit' has something to open."""
    os.makedirs(USER_DIR, exist_ok=True)
    for path, template in ((USER_VOCAB, USER_VOCAB_TEMPLATE), (USER_CORRECTIONS, USER_CORRECTIONS_TEMPLATE)):
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(template)


def build_prompt():
    """Glossary prompt for Whisper's initial_prompt, or None if there are no terms."""
    seen, terms = set(), []
    for path in (USER_VOCAB, os.path.join(HERE, "vocab.txt")):
        for term in _lines(path):
            if term.lower() not in seen:
                seen.add(term.lower())
                terms.append(term)
    prompt = "Glossary:"
    for term in terms:
        extra = f" {term},"
        if len(prompt) + len(extra) > PROMPT_MAX_CHARS:
            break
        prompt += extra
    return None if prompt == "Glossary:" else prompt.rstrip(",") + "."


def is_prompt_echo(text, prompt):
    """True if Whisper just parroted part of the prompt (a known hallucination on near-silence)."""
    if not prompt:
        return False

    def norm(s):
        return re.sub(r"[^a-z0-9 ]+", "", s.lower().replace("-", " "))

    t = norm(text).split()
    return len(t) >= 3 and " ".join(t) in " ".join(norm(prompt).split())


class Corrections:
    """Loads corrections from both files, reloading whenever either changes on disk."""

    def __init__(self):
        self._stamp = None
        self._rules = []

    def _paths(self):
        return (USER_CORRECTIONS, os.path.join(HERE, "corrections.txt"))

    def _load(self):
        rules = []
        for path in self._paths():  # user rules first, so they win
            for line in _lines(path):
                heard, sep, written = line.partition("=>")
                heard, written = heard.strip(), written.strip()
                if not sep or not heard or not written:
                    continue
                words = [re.escape(w) for w in heard.split()]
                pattern = r"(?<![\w])" + r"[\s-]+".join(words) + r"(?![\w])"
                rules.append((re.compile(pattern, re.IGNORECASE), written))
        return rules

    def apply(self, text):
        stamp = tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in self._paths())
        if stamp != self._stamp:
            self._stamp, self._rules = stamp, self._load()
        for pattern, written in self._rules:
            text = pattern.sub(lambda _m, w=written: w, text)
        return text
