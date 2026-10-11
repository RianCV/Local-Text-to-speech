"""
app.py — the user interface (a small web page that runs on your own computer).

Start it with:   python app.py      (or double-click "Read Aloud.command")
Your browser opens at http://127.0.0.1:7860 — "127.0.0.1" means "this computer
only": nothing is reachable from the network.

How it works:

  Python (this file + tts.py)              Browser (the PLAYER_JS script below)
  ───────────────────────────              ────────────────────────────────────
  render_reader(): your text, with   ──►   shows it; each sentence is clickable
  every sentence wrapped in a <span>
                                           to read sentence N it asks Python:
  synthesize_sentence(): one         ◄──   server.synthesize_sentence({text, voice, speed})
  sentence → WAV clip                ──►   plays the clip, and meanwhile asks for
                                           N+1 and N+2 so they're ready in time

The highlight is moved by the *browser*, at the moment the audio player
reports that sentence N has actually started playing — so it can't drift away
from the voice. Clicking a sentence just means "play from N".
"""

import os

# Settings that must be in place *before* the libraries are imported, because
# they read them once at import time:
#   HF_HUB_OFFLINE=1  -> never contact Hugging Face; use the model already
#                        downloaded by `python tts.py`. (Run with
#                        HF_HUB_OFFLINE=0 python app.py to allow downloads.)
#   GRADIO_ANALYTICS_ENABLED=False -> Gradio sends no usage statistics.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import base64
import html
import time
import warnings

# PyTorch prints a few harmless "deprecated"/"dropout" notices while loading
# Kokoro; hide them so the terminal stays readable.
warnings.filterwarnings("ignore", module=r"torch\.")

import gradio as gr
import numpy as np

from tts import (DEFAULT_VOICE, ENGLISH_VOICES, LANG_CODES, get_pipeline, split_sentences,
                 synthesize, to_wav_bytes, voice_label)


# ---------------------------------------------------------------------------
# Python side
# ---------------------------------------------------------------------------

def render_reader(text: str, voice: str, speed: float, action: str = "show") -> str:
    """
    Build the HTML for the reading panel: the whole text, with every sentence
    wrapped in <span class="s"> so the browser can highlight it and make it
    clickable. The text *between* sentences (spaces, line breaks) is kept as-is.

    The current voice/speed and an `action` ride along as data-* attributes
    for the player script to read:
      action="play" -> start reading from the first sentence
      action="show" -> just display (text, voice or speed changed)
    data-run is a unique number so that pressing "Read aloud" twice on the same
    text still counts as a change.

    html.escape() makes sure characters like < or & in your text are shown
    as-is instead of being interpreted as HTML.
    """
    spans = split_sentences(text)
    if not spans:
        body = '<span class="placeholder">Your text will appear here.</span>'
    else:
        pieces, position = [], 0
        for start, end in spans:
            pieces.append(html.escape(text[position:start]))  # gap before the sentence
            pieces.append(f'<span class="s">{html.escape(text[start:end])}</span>')
            position = end
        pieces.append(html.escape(text[position:]))
        body = "".join(pieces)
    return (f'<div class="reader" data-voice="{html.escape(voice)}" data-speed="{speed}" '
            f'data-action="{action}" data-run="{time.time_ns()}">{body}</div>')


def show_text(text: str, voice: str, speed: float) -> str:
    """Text, voice or speed changed: refresh the panel (doesn't start reading)."""
    return render_reader(text, voice, speed, action="show")


def start_reading(text: str, voice: str, speed: float) -> str:
    """The "Read aloud" button: refresh the panel and tell the player to start."""
    if not split_sentences(text):
        gr.Warning("Please paste some text first.")
        return gr.skip()
    return render_reader(text, voice, speed, action="play")


def synthesize_sentence(request: dict) -> str:
    """
    Called *from the browser*: server.synthesize_sentence({text, voice, speed})
    in PLAYER_JS. (Gradio passes the browser's data as a single argument, so we
    send one object and read its fields by name.)

    Returns the sentence's audio as a "data URL" — the WAV file encoded as
    text — which the browser's audio player can play directly.
    """
    # The values come from the web page, so double-check them.
    sentence = str(request.get("text", ""))
    voice = request.get("voice")
    if voice not in ENGLISH_VOICES:
        voice = DEFAULT_VOICE
    speed = min(max(float(request.get("speed") or 1.0), 0.5), 2.0)

    wav = to_wav_bytes(synthesize(sentence, voice, speed))
    return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")


# ---------------------------------------------------------------------------
# Browser side: the reading panel's look, and the player script
# ---------------------------------------------------------------------------

# The panel = a small toolbar (pause button + status) followed by the text.
# ${value} is replaced by whatever render_reader() returned.
READER_TEMPLATE = """
<div class="toolbar">
  <button class="pause" type="button">▶ Play</button>
  <span class="status"></span>
</div>
${value}
"""

PLAYER_JS = r"""
// Gradio runs this once when the page loads. Available here:
//   element  = the panel's HTML element
//   server   = our Python functions (server.synthesize_sentence)
//   watch    = run a callback whenever Python sends new content for the panel

const audio = new Audio();   // one invisible audio player, reused for every sentence
const clips = new Map();     // audio already requested: "voice|speed|text" -> Promise of a data URL
let current = -1;            // index of the sentence being read (-1 = none)
let attempt = 0;             // increases on every jump/stop, so outdated work can be ignored
let statusText = 'Press "Read aloud", or click any sentence to start there.';
let settings = { voice: '', speed: 1 };
let lastText = null;

const sentences = () => [...element.querySelectorAll('.reader .s')];

// Safari (and others) only let a page start sound *during* a click. Our audio
// arrives a moment after the click (Python has to generate it first), so on
// your first click anywhere on the page we play a tiny silent clip on our one
// audio player. From then on the browser trusts that player.
let unlocked = false;
document.addEventListener('click', () => {
  if (unlocked) return;
  unlocked = true;
  audio.src = '__SILENT_WAV__';
  audio.play().catch((e) => { if (e.name === 'NotAllowedError') unlocked = false; });
}, true);

function setStatus(message) {
  if (message !== undefined) statusText = message;
  const el = element.querySelector('.status');
  if (el) el.textContent = statusText;
  const button = element.querySelector('.pause');
  if (button) button.textContent = (current >= 0 && !audio.paused) ? '⏸ Pause' : '▶ Play';
}

function highlight(index, scroll = true) {
  sentences().forEach((s, k) => s.classList.toggle('current', k === index));
  const s = sentences()[index];
  const box = element.querySelector('.reader');
  if (s && box && scroll) {
    // Scroll the panel (not the whole page) so the sentence sits in the middle.
    box.scrollTo({ top: s.offsetTop - box.clientHeight / 2, behavior: 'smooth' });
  }
}

// Ask Python for the audio of sentence `index` (or reuse an earlier request).
function clipFor(index) {
  const text = sentences()[index].textContent;
  const key = `${settings.voice}|${settings.speed}|${text}`;
  if (!clips.has(key)) {
    clips.set(key, server.synthesize_sentence({ text, voice: settings.voice, speed: settings.speed }));
  }
  return clips.get(key);
}

// Play sentence `index`, then automatically continue with the next ones.
async function playFrom(index) {
  const myAttempt = ++attempt;
  audio.pause();
  const total = sentences().length;
  if (index >= total) {
    current = -1;
    highlight(-1, false);
    setStatus('Finished ✓  Click any sentence to hear it again.');
    return;
  }
  current = index;
  setStatus(`Preparing sentence ${index + 1} of ${total}…`);

  let url;
  try {
    url = await clipFor(index);
  } catch (error) {
    if (myAttempt === attempt) setStatus('Error while generating audio: ' + error);
    return;
  }
  if (myAttempt !== attempt) return;  // you clicked somewhere else in the meantime

  // While this sentence plays, have the next two prepared.
  for (const next of [index + 1, index + 2]) {
    if (next < total) clipFor(next).catch(() => {});
  }

  audio.src = url;
  // 'playing' fires when sound actually starts coming out: that is the moment
  // to move the highlight, so it matches the voice exactly.
  audio.onplaying = () => {
    if (myAttempt !== attempt) return;
    highlight(index);
    setStatus(`Reading sentence ${index + 1} of ${total}`);
  };
  audio.onended = () => { if (myAttempt === attempt) playFrom(index + 1); };
  try {
    await audio.play();
  } catch (error) {
    // Browsers may block sound until you've clicked on the page once.
    setStatus('Click the sentence (or ▶ Play) to start the sound.');
  }
}

function stop(message) {
  attempt++;
  audio.pause();
  current = -1;
  highlight(-1, false);
  setStatus(message);
}

// One click handler for the whole panel (it keeps working when the text is replaced).
element.addEventListener('click', (event) => {
  if (event.target.closest('.pause')) {
    if (current < 0) playFrom(0);                              // nothing started yet
    else if (audio.paused) { audio.play(); setStatus(); }      // resume
    else { audio.pause(); setStatus('Paused'); }               // pause
    return;
  }
  const sentence = event.target.closest('.reader .s');
  if (sentence) playFrom(sentences().indexOf(sentence));
});

// Python sent new panel content (Read aloud pressed, or text/voice/speed changed).
function onNewContent() {
  const reader = element.querySelector('.reader');
  if (!reader) return;
  settings = { voice: reader.dataset.voice, speed: parseFloat(reader.dataset.speed) };
  const text = sentences().map(s => s.textContent).join('\n');
  const textChanged = text !== lastText;
  lastText = text;
  if (textChanged) clips.clear();  // old audio no longer matches anything

  if (reader.dataset.action === 'play') {
    playFrom(0);
  } else if (textChanged) {
    stop('Press "Read aloud", or click any sentence to start there.');
  } else {
    // Only voice/speed changed: keep reading. The next sentence uses the new
    // settings (its audio is requested with them). Restore the highlight,
    // since the panel was redrawn.
    highlight(current, false);
    setStatus();
  }
}

watch('value', onNewContent);
onNewContent();
"""
# 0.1 s of silence as a data URL, for the "unlock" trick at the top of the script.
SILENT_WAV = "data:audio/wav;base64," + base64.b64encode(to_wav_bytes(np.zeros(2400))).decode()
PLAYER_JS = PLAYER_JS.replace("__SILENT_WAV__", SILENT_WAV)

CSS = """
#reader .toolbar { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
#reader .pause {
    padding: 0.3rem 0.9rem; border-radius: 6px; cursor: pointer;
    border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary);
    color: var(--body-text-color);
}
#reader .status { font-size: 0.9rem; opacity: 0.75; }
#reader .reader {
    white-space: pre-wrap;          /* keep your line breaks and paragraphs */
    font-size: 1.1rem;
    line-height: 1.7;
    max-height: 65vh;               /* long texts scroll inside the panel */
    overflow-y: auto;
    position: relative;             /* lets the script measure sentence positions */
    padding: 0.5rem 0.25rem;
}
#reader .s { cursor: pointer; border-radius: 3px; }
#reader .s:hover { text-decoration: underline dotted; }
#reader .s.current { background: #ffe58a; color: #1a1a1a; }
#reader .placeholder { opacity: 0.5; }
"""


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Read Aloud") as demo:
    gr.Markdown("# Read Aloud\nPaste your text, pick a voice, and press **Read aloud**. "
                "Click any sentence to continue reading from there.")

    with gr.Row():
        # Left column: input and controls
        with gr.Column():
            text_in = gr.Textbox(label="Your text", lines=16,
                                 placeholder="Paste or type the text to read...")
            voice_in = gr.Dropdown(
                # Each choice is (label shown, value passed to our functions).
                choices=[(voice_label(v), v) for v in ENGLISH_VOICES],
                value=DEFAULT_VOICE, label="Voice")
            speed_in = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
            read_btn = gr.Button("Read aloud", variant="primary")

        # Right column: follow along (the panel contains the player)
        with gr.Column():
            reader = gr.HTML(
                render_reader("", DEFAULT_VOICE, 1.0),
                html_template=READER_TEMPLATE,
                js_on_load=PLAYER_JS,
                server_functions=[synthesize_sentence],
                elem_id="reader",
            )

    inputs = [text_in, voice_in, speed_in]
    read_btn.click(start_reading, inputs=inputs, outputs=reader)
    # Keep the panel in sync with the controls. Voice/speed changes apply from
    # the next sentence on, without interrupting the reading.
    text_in.change(show_text, inputs=inputs, outputs=reader)
    voice_in.change(show_text, inputs=inputs, outputs=reader)
    speed_in.release(show_text, inputs=inputs, outputs=reader)


if __name__ == "__main__":
    # Load the model and the voices' accent now, and generate one throwaway word:
    # PyTorch's very first run in a fresh process is slow (several seconds),
    # so we pay that cost here instead of on your first click.
    print("Loading Kokoro...")
    for lang_code in LANG_CODES:
        get_pipeline(lang_code)
    synthesize("Ready.")
    demo.launch(server_name="127.0.0.1", inbrowser=True, css=CSS)
