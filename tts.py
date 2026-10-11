"""
tts.py — the speech engine (no UI code here).

This file wraps the Kokoro-82M text-to-speech model:
  * loads the model once and keeps it in memory,
  * picks the right pronunciation rules (American or British) for each voice,
  * splits text into sentences so the UI can highlight what is being read,
  * turns one sentence at a time into a WAV clip for the browser to play.

Run it directly once after installing, while online:

    python tts.py

That downloads the model + all English voices into the local Hugging Face
cache (~/.cache/huggingface) and writes a short test clip to sample.wav.
After that, the app never needs the internet.
"""

import functools
import io
import re
import threading
import time

import numpy as np
import soundfile as sf
from kokoro import KModel, KPipeline

# Where Kokoro's weights and voices live on Hugging Face (downloaded once, then cached).
REPO_ID = "hexgrad/Kokoro-82M"

# Kokoro always produces audio at 24,000 samples per second.
SAMPLE_RATE = 24000

DEFAULT_VOICE = "am_echo"

# The voices offered in the app. Kokoro voice names encode two things:
#   1st letter = accent:  a = American, b = British
#   2nd letter = gender:  f = female,   m = male
# The voice only controls the *timbre*; the accent letter also decides which
# pronunciation rules we use (see get_pipeline below).
# Kokoro has 28 English voices (e.g. "af_heart", "bf_emma", "bm_george"); see
# https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md — add any of
# them here to offer it, then run `python tts.py` once to download it.
ENGLISH_VOICES = ["am_echo", "am_eric"]

# Accents used by the voices above ("a" and/or "b"), so we only load what's needed.
LANG_CODES = sorted({voice[0] for voice in ENGLISH_VOICES})

_GENDERS = {"f": "female", "m": "male"}


def voice_label(voice: str) -> str:
    """Human-friendly name for the dropdown, e.g. 'am_echo' -> 'Echo (male)'."""
    gender, name = voice[1], voice[3:]
    return f"{name.capitalize()} ({_GENDERS[gender]})"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# @functools.cache means "run this function once, then return the same result
# every time". So the model is loaded on first use and reused afterwards.

@functools.cache
def get_model() -> KModel:
    """Load the 82M-parameter Kokoro model on the CPU (takes a few seconds)."""
    return KModel(repo_id=REPO_ID).to("cpu").eval()  # eval() = inference mode


@functools.cache
def get_pipeline(lang_code: str) -> KPipeline:
    """
    A KPipeline turns text into phonemes (pronunciation) and feeds them to the model.

    Kokoro separates pronunciation from voice: lang_code 'a' uses American rules
    ("tomato" = to-MAY-to), 'b' uses British rules. We create at most two
    pipelines, and both share the *same* model object, so the ~330 MB of weights
    are only loaded once.
    """
    return KPipeline(lang_code=lang_code, repo_id=REPO_ID, model=get_model())


def pipeline_for_voice(voice: str) -> KPipeline:
    # The voice's first letter is its accent, which is exactly Kokoro's lang_code.
    return get_pipeline(voice[0])


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
# A sentence = starts at a non-space character and runs (as little as possible)
# until one of:
#   * sentence-ending punctuation (. ! ? or several, e.g. "?!" or "..."),
#     optionally followed by closing quotes/brackets, and then whitespace or
#     end of text. Requiring whitespace after the dot keeps "3.14" together.
#   * the end of a line (so titles/headings without a period are their own piece),
#   * the end of the text.
_SENTENCE_RE = re.compile(
    r"""\S[^\n]*?                         # start at a non-space, take as little as possible...
        (?: [.!?]+["'”’)\]]*(?=\s|$)      # ...up to end punctuation followed by space/end
          | (?=\n)                        # ...or up to the end of the line
          | $ )                           # ...or up to the end of the text
    """,
    re.VERBOSE,
)

# The regex alone would also split after abbreviations ("Dr. | Smith"), which
# makes Kokoro pause as if the sentence ended. A piece ending in one of these
# (or in a single-letter initial like "J.") gets glued to the next piece.
# ("etc." and "No." are left out on purpose: they often really end a sentence.)
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.",
                  "vs.", "e.g.", "i.e.", "fig.", "approx."}
_INITIAL_RE = re.compile(r"(?:^|\s)[A-Z]\.$")


def _ends_with_abbreviation(piece: str) -> bool:
    last_word = piece.rsplit(maxsplit=1)[-1].lower()
    return last_word in _ABBREVIATIONS or bool(_INITIAL_RE.search(piece))


def split_sentences(text: str) -> list[tuple[int, int]]:
    """
    Return (start, end) character positions of each sentence in `text`.

    We keep *positions* rather than copies of the sentences so the UI can
    highlight the exact spot in the original text (with its line breaks intact).
    """
    spans: list[tuple[int, int]] = []
    for start, end in (m.span() for m in _SENTENCE_RE.finditer(text)):
        if spans:
            prev_start, prev_end = spans[-1]
            same_line = "\n" not in text[prev_end:start]
            if same_line and _ends_with_abbreviation(text[prev_start:prev_end]):
                spans[-1] = (prev_start, end)  # merge with the previous piece
                continue
        spans.append((start, end))
    return spans


# ---------------------------------------------------------------------------
# Speech generation
# ---------------------------------------------------------------------------

# The browser asks for several sentences at once (the one to play now plus the
# next ones, prepared in advance). Those requests arrive on different threads,
# but the model should only work on one sentence at a time — this lock makes
# the other requests wait their turn.
_model_lock = threading.Lock()


def synthesize(sentence: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> np.ndarray:
    """
    Turn one sentence into speech.

    Returns a float32 numpy array at SAMPLE_RATE with values in [-1, 1]
    (empty if there was nothing pronounceable, e.g. just "—").
    """
    pipeline = pipeline_for_voice(voice)
    pieces = []
    with _model_lock:
        # split_pattern=None: we already split the text into sentences ourselves.
        # Kokoro may still cut a *very* long sentence into smaller chunks (its
        # limit is ~510 phonemes), so we glue those pieces back together.
        for result in pipeline(sentence, voice=voice, speed=speed, split_pattern=None):
            if result.audio is not None:
                pieces.append(result.audio.numpy())
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def to_wav_bytes(audio: np.ndarray) -> bytes:
    """
    Package audio as a standard WAV file (16-bit), in memory, ready to send
    to the browser.
    """
    int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    sf.write(buffer, int16, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# One-time setup + smoke test:  python tts.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading model (downloads ~330 MB the first time)...")
    for lang_code in LANG_CODES:
        get_pipeline(lang_code)

    print(f"Downloading {len(ENGLISH_VOICES)} English voices...")
    for voice in ENGLISH_VOICES:
        pipeline_for_voice(voice).load_voice(voice)

    sample = (
        "Hello! This is Kokoro, reading your text aloud on this computer. "
        "Everything runs offline, on the CPU."
    )
    t0 = time.perf_counter()
    audio = np.concatenate([synthesize(sample[s:e]) for s, e in split_sentences(sample)])
    elapsed = time.perf_counter() - t0
    duration = len(audio) / SAMPLE_RATE

    sf.write("sample.wav", audio, SAMPLE_RATE)
    print(f"Wrote sample.wav: {duration:.1f}s of audio generated in {elapsed:.1f}s "
          f"(real-time factor {elapsed / duration:.2f}; below 1.0 = faster than real time)")
