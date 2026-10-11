#!/bin/bash
# Double-click this file in Finder to start the reader.
# It opens a Terminal window (keep it open while you use the app; close it to quit)
# and your browser shows the app at http://127.0.0.1:7860
cd "$(dirname "$0")"
source .venv/bin/activate
python app.py
