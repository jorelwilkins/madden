# 🎮 Madden 26 Highlight + Shorts Agent

A Python agent that automatically creates **highlights and YouTube Shorts** from Madden NFL 26 gameplay videos.

## What It Does
- Generates **one 10-minute highlight** per video
- Generates **two vertical YouTube Shorts**
- Creates a **YouTube-ready metadata CSV** (title, description, tags)
- Watches a folder and processes new videos automatically

Optimized for **Madden NFL 26 (PS5)** using broadcast presentation.

## Requirements
- macOS (Mac Studio tested)
- Python 3.10+
- FFmpeg
- Google Drive for Desktop

## Install
```bash
brew install ffmpeg
pip3 install watchdog opencv-python numpy

## Configure
WATCH_FOLDER = "/Users/YOURNAME/Library/CloudStorage/GoogleDrive-EMAIL/My Drive/game video"

Run
source .venv/bin/activate
python3 highlight_watcher_plus_shorts.py
