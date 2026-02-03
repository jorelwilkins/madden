import time
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import csv
import json

import cv2
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ====================
# SETTINGS (EDIT THESE)
# ====================

# ✅ CHANGE THIS to your Google Drive local folder 
# Example:
# WATCH_FOLDER = "/Users/yourname/Library/CloudStorage/GoogleDrive-your@gmail.com/My Drive/game video"
WATCH_FOLDER = "/Users/jd/stream-agents/madden26-agent"

VIDEO_EXTS = {".mp4", ".mkv", ".mov"}

# Output naming
HIGHLIGHT_SUFFIX = "_BEST.mp4"
SHORT_SUFFIX_FMT = "_SHORT_{:02d}.mp4"   # _SHORT_01, _SHORT_02

# Durations
HIGHLIGHT_LEN_SEC = 600    # 10 minutes
SHORT_LEN_SEC = 45         # Shorts
SHORTS_TO_MAKE = 2

# Motion sampling
SAMPLE_SEC = 0.5           # sample frames every 0.5 sec
PRE_ROLL_HIGHLIGHT = 5     # seconds before highlight start
PRE_ROLL_SHORT = 3         # seconds before short start

# Avoid overlapping shorts
MIN_GAP_SHORT_SEC = 60

# Wait for downloads to finish
STABLE_SECONDS = 10
CHECK_INTERVAL = 2

# Shorts formatting (vertical)
MAKE_VERTICAL_SHORTS = True

# YouTube metadata defaults (edit once)
GAME_NAME = "Madden NFL 26"
PLATFORM = "PS5"
HASHTAGS = ["#madden26", "#mut", "#ps5", "#gaming", "#shorts"]
TAGS = ["madden 26", "madden ultimate team", "mut", "h2h", "ps5", "madden highlights", "gaming"]
LABELS = ["madden26", "mut", "ps5"]
CATEGORY = "Gaming"
PRIVACY_SUGGESTION = "public"


# ====================
# FFmpeg helpers
# ====================
def run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def ffprobe_duration(video_path: str) -> float:
    out = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ])
    return float(out.strip())

def export_clip_reencode(video_path: str, out_path: str, start_sec: float, duration_sec: float, vertical: bool = False):
    start_sec = max(0.0, float(start_sec))
    duration_sec = max(1.0, float(duration_sec))

    vf = None
    if vertical:
        # Center-crop to 9:16 based on height, then scale to 1080x1920
        vf = "crop=in_h*9/16:in_h:(in_w-in_h*9/16)/2:0,scale=1080:1920"

    cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", video_path, "-t", str(duration_sec)]
    if vf:
        cmd += ["-vf", vf]

    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        out_path
    ]
    run(cmd)


# ====================
# File stability check
# ====================
def wait_until_file_stable(path: Path, stable_seconds=STABLE_SECONDS) -> bool:
    last_size = -1
    stable_for = 0
    while stable_for < stable_seconds:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False

        if size == last_size and size > 0:
            stable_for += CHECK_INTERVAL
        else:
            stable_for = 0
            last_size = size

        time.sleep(CHECK_INTERVAL)
    return True


# ====================
# Motion scoring
# ====================
def compute_motion_series(video_path: str, sample_sec: float) -> Tuple[np.ndarray, np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (total_frames / fps) if total_frames else ffprobe_duration(video_path)

    step = max(1, int(sample_sec * fps))
    times, motions = [], []
    prev = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if prev is not None:
                diff = cv2.absdiff(prev, gray)
                motions.append(float(np.mean(diff)))
                times.append(frame_idx / fps)
            prev = gray

        frame_idx += 1

    cap.release()

    if len(motions) < 5:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32), float(duration)

    return np.array(times, dtype=np.float32), np.array(motions, dtype=np.float32), float(duration)


def best_motion_window_start(times: np.ndarray, motions: np.ndarray, duration: float, window_sec: int) -> float:
    if duration <= window_sec:
        return 0.0

    best_score = -1.0
    best_start = 0.0

    j = 0
    running_sum = 0.0
    count = 0

    for i in range(len(times)):
        while j < len(times) and times[j] <= times[i] + window_sec:
            running_sum += motions[j]
            count += 1
            j += 1

        avg = (running_sum / count) if count else 0.0
        if avg > best_score:
            best_score = avg
            best_start = float(times[i])

        running_sum -= motions[i]
        count -= 1

    start = max(0.0, best_start - PRE_ROLL_HIGHLIGHT)
    start = min(start, max(0.0, duration - window_sec))
    return start


def pick_top_motion_peaks(times: np.ndarray, motions: np.ndarray, k: int, min_gap_sec: int, duration: float) -> List[float]:
    idx_sorted = np.argsort(-motions)
    peaks = []
    for idx in idx_sorted:
        t = float(times[idx])
        if all(abs(t - p) > min_gap_sec for p in peaks):
            t = min(t, max(0.0, duration - SHORT_LEN_SEC))
            peaks.append(t)
        if len(peaks) >= k:
            break
    return sorted(peaks)


# ====================
# Metadata
# ====================
def build_metadata(kind: str, start_sec: float, duration_sec: int) -> dict:
    if kind == "highlight":
        title = f"{GAME_NAME} Highlights (Best 10 Minutes) | {PLATFORM}"
    else:
        title = f"{GAME_NAME} Crazy Moment 😤 | {PLATFORM}"

    description = "\n".join([
        f"{GAME_NAME} on {PLATFORM}.",
        "More highlights coming — follow for more.",
        "",
        "Tags: " + ", ".join(TAGS),
        "",
        " ".join(HASHTAGS),
    ])

    return {
        "title": title[:95],
        "description": description,
        "tags": TAGS,
        "labels": LABELS + [kind],
        "category": CATEGORY,
        "privacy_suggestion": PRIVACY_SUGGESTION,
        "start_sec": round(float(start_sec), 2),
        "duration_sec": int(duration_sec),
    }


def write_metadata_files(out_dir: Path, rows: List[dict]):
    csv_path = out_dir / "youtube_metadata.csv"
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "kind", "title", "description", "tags", "labels", "category", "privacy_suggestion", "start_sec", "duration_sec"
        ])
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({
                "file": r["file"],
                "kind": r["kind"],
                "title": r["meta"]["title"],
                "description": r["meta"]["description"],
                "tags": ", ".join(r["meta"]["tags"]),
                "labels": ", ".join(r["meta"]["labels"]),
                "category": r["meta"]["category"],
                "privacy_suggestion": r["meta"]["privacy_suggestion"],
                "start_sec": r["meta"]["start_sec"],
                "duration_sec": r["meta"]["duration_sec"],
            })

    json_path = out_dir / "youtube_metadata.json"
    existing = []
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except:
            existing = []
    existing.extend(rows)
    json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


# ====================
# Processing
# ====================
def output_highlight_path(video_path: Path) -> Path:
    return video_path.with_name(video_path.stem + HIGHLIGHT_SUFFIX)

def output_short_path(video_path: Path, idx: int) -> Path:
    return video_path.with_name(video_path.stem + SHORT_SUFFIX_FMT.format(idx))

def should_process(video_path: Path) -> bool:
    if video_path.suffix.lower() not in VIDEO_EXTS:
        return False
    if video_path.name.startswith("."):
        return False

    highlight_exists = output_highlight_path(video_path).exists()
    shorts_exist = all(output_short_path(video_path, i).exists() for i in range(1, SHORTS_TO_MAKE + 1))
    return not (highlight_exists and shorts_exist)

def process_video(video_path: Path):
    try:
        print(f"\n[NEW] {video_path.name}")
        print("[WAIT] Checking file stability...")
        if not wait_until_file_stable(video_path):
            print("[SKIP] File disappeared before stable.")
            return

        out_dir = video_path.parent
        duration = ffprobe_duration(str(video_path))
        print(f"[INFO] Duration: {int(duration)} sec")

        times, motions, duration = compute_motion_series(str(video_path), SAMPLE_SEC)

        meta_rows = []

        # 1) Best 10-minute highlight
        highlight_out = output_highlight_path(video_path)
        if not highlight_out.exists():
            start = best_motion_window_start(times, motions, duration, HIGHLIGHT_LEN_SEC)
            print(f"[HIGHLIGHT] Start @ {start:.2f}s -> {highlight_out.name}")
            export_clip_reencode(str(video_path), str(highlight_out), start, min(HIGHLIGHT_LEN_SEC, duration), vertical=False)

            meta_rows.append({
                "file": highlight_out.name,
                "kind": "highlight",
                "meta": build_metadata("highlight", start, min(HIGHLIGHT_LEN_SEC, int(duration))),
            })
        else:
            print(f"[SKIP] Highlight exists: {highlight_out.name}")

        # 2) Shorts (top motion peaks)
        peaks = pick_top_motion_peaks(times, motions, SHORTS_TO_MAKE, MIN_GAP_SHORT_SEC, duration)
        for i, peak_t in enumerate(peaks, start=1):
            short_out = output_short_path(video_path, i)
            if short_out.exists():
                print(f"[SKIP] Short exists: {short_out.name}")
                continue

            start = max(0.0, peak_t - PRE_ROLL_SHORT)
            start = min(start, max(0.0, duration - SHORT_LEN_SEC))
            print(f"[SHORT {i}] Peak @ {peak_t:.2f}s, start @ {start:.2f}s -> {short_out.name}")

            export_clip_reencode(str(video_path), str(short_out), start, min(SHORT_LEN_SEC, duration), vertical=MAKE_VERTICAL_SHORTS)

            meta_rows.append({
                "file": short_out.name,
                "kind": "short",
                "meta": build_metadata("short", start, min(SHORT_LEN_SEC, int(duration))),
            })

        if meta_rows:
            write_metadata_files(out_dir, meta_rows)
            print("[META] Wrote youtube_metadata.csv + youtube_metadata.json")

        print("[DONE] ✅")

    except Exception as e:
        print(f"[ERROR] {video_path.name}: {e}")


# ====================
# Watcher
# ====================
class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if should_process(p):
            process_video(p)

    def on_moved(self, event):
        if event.is_directory:
            return
        p = Path(event.dest_path)
        if should_process(p):
            process_video(p)

def initial_scan(folder: Path):
    print(f"[SCAN] {folder}")
    for p in sorted(folder.iterdir()):
        if p.is_file() and should_process(p):
            process_video(p)

def main():
    folder = Path(WATCH_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)

    initial_scan(folder)

    print(f"\n[WATCHING] {WATCH_FOLDER}")
    print(f"Creates: {HIGHLIGHT_SUFFIX} + {SHORTS_TO_MAKE} shorts per video")
    print("Stop with Ctrl+C\n")

    observer = Observer()
    observer.schedule(VideoHandler(), WATCH_FOLDER, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] watcher stopped.")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
