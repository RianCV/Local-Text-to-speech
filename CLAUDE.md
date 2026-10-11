# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal, offline, CPU-only text-to-speech reader. It has two Python files and no build step: `tts.py` (the Kokoro-82M engine) and `app.py` (the Gradio 6 UI plus an in-browser player written in JS). The owner wants the code **simple and heavily commented** so they can understand every part, and wants each key decision explained. Keep that style, and keep the comment density of the existing code.

## Commands

The venv **must** use Python 3.10–3.12. `kokoro==0.9.4` does not install on 3.13, which is the machine's pyenv default. The venv was built from `/opt/homebrew/bin/python3.12`.

```bash
source .venv/bin/activate
pip install -r requirements.txt   # pinned versions; spaCy's en_core_web_sm is a wheel URL in the file
python tts.py                     # one-time download of model + voices (needs internet); smoke test → sample.wav + real-time factor
python app.py                     # serves http://127.0.0.1:7860 and opens the browser
```

`Read Aloud.command` is the double-click launcher the owner uses every day. It does the same thing as `python app.py`.

There is no test suite and no linter. To verify changes:
- **Engine:** run `python tts.py`. To check `split_sentences()` edge cases, call it directly (headings, "3.14", "?!", "...", quotes, "Dr."/"e.g."/initials, empty input).
- **Server without a browser:** `import app`, then call `app.synthesize_sentence({...})` / `app.render_reader(...)`.
- **Player behavior** (highlight sync, click-to-jump, pause, voice switching) can only be checked in a real browser engine. Earlier it was tested with Playwright (WebKit + Chromium), installed outside the repo. To do the same, start the server with `inbrowser=False` on another port and set `BROWSER=true` so no tab opens on the owner's screen. The owner uses **Safari**.

## Architecture

**Playback is driven by the browser, not by Python.** An earlier version streamed audio from a Python generator through `gr.Audio(streaming=True)` and paced the highlight with server-side timing. The highlight drifted because the browser buffers on its own schedule, so that design was replaced. Don't reintroduce server-side pacing or streaming output.

The flow:
1. `render_reader()` (Python) turns the text into HTML. Every sentence from `tts.split_sentences()` is wrapped in `<span class="s">`, and the text between sentences is kept verbatim (`white-space: pre-wrap`). `data-voice`, `data-speed`, `data-action` (`play` | `show`) and a `data-run` nonce sit on the `.reader` div.
2. That HTML is the value of a `gr.HTML` component with `html_template`, `js_on_load=PLAYER_JS` and `server_functions=[synthesize_sentence]`. The Read button, text change, voice change and speed release all re-render it. `watch('value', ...)` in the JS decides what to do:
   - `play` restarts from sentence 0
   - changed text stops playback
   - a voice/speed-only change keeps reading, and the change applies from the next sentence
3. The JS keeps a single `Audio` element. For sentence N it calls `server.synthesize_sentence({text, voice, speed})`, which returns a `data:audio/wav;base64` URL, and prefetches N+1 and N+2. Clips are cached by `voice|speed|text`. The highlight moves in `audio.onplaying`, which is why it stays in sync. `onended` chains to the next sentence. An `attempt` counter discards stale async work after a jump or stop.

The JS identifies sentences by their **span index and `textContent`**, so `split_sentences()` determines both the highlight granularity and what gets sent to the model.

## Non-obvious gotchas (verified against the installed versions)

- **Gradio 6 `server_functions` receive ONE argument.** The frontend sends a single JS argument as-is, and packs multiple arguments into one array. That is why the JS passes one object and the Python function takes a `dict`.
- **Gradio 6 moved `css`/`theme`/`js` from `gr.Blocks()` to `demo.launch()`.**
- **Safari autoplay:** audio arrives after an async server call, outside the click gesture. A capture-phase document click listener "unlocks" the single `Audio` element by playing `__SILENT_WAV__`, which is substituted into `PLAYER_JS` from Python. Keep reusing one `Audio` element; creating new ones would lose the unlock.
- **Model access is serialized:** server functions run concurrently in a thread pool, so `tts.synthesize()` holds `_model_lock`. One `KModel` (CPU) is shared by the `KPipeline`s, which are created per lang code and cached with `functools.cache`.
- **The voice name's first letter is Kokoro's `lang_code`** (`a` = American and `b` = British pronunciation). `LANG_CODES` is derived from `ENGLISH_VOICES`, and only those pipelines are preloaded. The owner chose to offer only `am_echo` (default) and `am_eric`. Adding a voice also requires running `python tts.py` once to download it.
- **Offline by design:** `app.py` sets `HF_HUB_OFFLINE=1` and `GRADIO_ANALYTICS_ENABLED=False` *before* importing the libraries, because they read these at import time. A voice that isn't already in the HF cache fails in the app. Download it with `python tts.py` (or `HF_HUB_OFFLINE=0 python app.py`).
- **Audio is converted to int16 WAV in `to_wav_bytes()`.** Gradio peak-normalizes float audio per chunk.
- **espeak-ng:** misaki uses the copy bundled by `espeakng-loader`, not the Homebrew install.
- **Startup warm-up:** `app.py` runs `synthesize("Ready.")` before launching, because the first PyTorch inference in a fresh process takes about 7 s.
