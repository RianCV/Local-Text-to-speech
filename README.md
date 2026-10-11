# Local Text-to-Speech Reader

A small app that reads your text aloud with a natural-sounding voice. It runs
entirely on your own computer: no internet connection (after setup), no
accounts, no paid APIs.

Paste a text, press **Read aloud**, and follow along. The sentence being spoken
is highlighted, and you can click any sentence to continue reading from there.

## Why I made this

This is a personal project. I built it to help me read essays, because I enjoy
reading along with a voice: hearing the text while following it on screen
helps me stay focused and get through long pieces. I wanted something simple
and private that runs on my own machine, so I made this.

## Features

- Paste or type any English text.
- The sentence being spoken is highlighted as the voice reads it.
- Click any sentence to jump there.
- Pause and resume.
- Two voices, **Echo** and **Eric**, and adjustable speed (0.5× to 2×).
- Long texts work fine: audio starts after the first sentence instead of
  waiting for the whole text.
- Runs on a normal CPU (no graphics card needed) and works offline after the
  first setup.

The voices come from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
a small open-source text-to-speech model (Apache 2.0 license).

## How to run it

I built and tested this on a Mac (Apple Silicon). It should also work on
Windows and Linux, but I haven't tried those.

You'll need about 1.5 GB of free disk space and an internet connection for
the first setup only.

### 1. Install Python 3.12

Kokoro needs Python **3.10, 3.11 or 3.12** (3.13 is not supported yet).

- **Mac:** `brew install python@3.12`, or download it from [python.org](https://www.python.org/downloads/)
- **Windows:** download Python 3.12 from [python.org](https://www.python.org/downloads/). During install, tick "Add Python to PATH".
- **Linux:** use your package manager, e.g. `sudo apt install python3.12 python3.12-venv`

### 2. Download this project

```bash
git clone https://github.com/RianCV/Local-Text-to-speech.git
cd Local-Text-to-speech
```

(Or click **Code → Download ZIP** on GitHub and unzip it.)

### 3. Create a virtual environment and install the requirements

This keeps the project's packages separate from the rest of your computer.

**Mac / Linux:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

This downloads PyTorch and a few other packages, so it can take several minutes.

### 4. Download the voice model (one time only)

```bash
python tts.py
```

This downloads the Kokoro model (~330 MB) and the voices, then saves a short
test clip called `sample.wav`. Play the clip to check that everything works.
After this step, the app no longer needs the internet.

### 5. Start the app

```bash
python app.py
```

Your browser opens at `http://127.0.0.1:7860`. The app is only reachable from
your own computer. Paste your text, pick a voice and press **Read aloud**.
To stop the app, close the terminal window or press `Ctrl+C`.

**Next time**, you only need two things: activate the environment (the
`activate` line from step 3), then run `python app.py`. On a Mac you can
simply double-click **`Read Aloud.command`** instead.

## How it works (short version)

- `tts.py` is the speech engine. It loads Kokoro, splits your text into
  sentences and turns each sentence into audio.
- `app.py` is the web page, built with [Gradio](https://www.gradio.app/). A
  small script in the page plays one sentence at a time and asks for the next
  ones in advance. The highlight moves at the exact moment each sentence starts
  playing, which keeps it in sync with the voice.

## Troubleshooting

- **`pip install` fails mentioning `kokoro` or the Python version:** you are
  probably on Python 3.13. Recreate the environment with Python 3.12 (step 3).
- **No sound on the first try in Safari:** click a sentence or the ▶ Play
  button once. Browsers only allow sound to start after you've clicked on the
  page.
- **Want more voices?** Kokoro has 28 English voices. Add their names to
  `ENGLISH_VOICES` in `tts.py` (see the
  [voice list](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)),
  then run `python tts.py` once to download them.
