# Madden Highlight + Shorts Watcher

This script watches a folder for new videos and automatically creates:
- a 10-minute "best highlight" clip (`*_BEST.mp4`)
- 2 vertical shorts (`*_SHORT_01.mp4`, `*_SHORT_02.mp4`)
- YouTube metadata files (`youtube_metadata.csv` + `youtube_metadata.json`)

---

## Requirements (both Windows + macOS)

### 1) Python 3.10+
Check:
- `python --version`

### 2) FFmpeg (must include `ffmpeg` AND `ffprobe`)
Check:
- `ffmpeg -version`
- `ffprobe -version`

### 3) Python packages
- `opencv-python`
- `numpy`
- `watchdog`

---

## Recommended: Make WATCH_FOLDER configurable (one small edit)

Right now the script hardcodes:
```py
WATCH_FOLDER = "/Users/jd/stream-agents/madden26-agent"
