"""
Learn where subtitle/caption text appears in this video by sampling frames.
Detection-only pass — no recognition, just box positions.
"""

from pathlib import Path

import cv2
import numpy as np


def learn_roi(
    video_path: Path,
    reader,          # easyocr.Reader, passed in to avoid double-loading
    n_samples: int = 50,
) -> tuple[int, int, int, int] | None:
    """
    Sample n_samples frames spread across the video, collect text box vertical
    positions, check whether ≥60% cluster in the top or bottom 30% of the frame.

    Returns (x1, y1, x2, y2) of the dominant text band, or None if text is
    scattered (→ caller should fall back to full-frame OCR).
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if total <= 0 or h <= 0 or w <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, n_samples, dtype=int)
    y_centers: list[float] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        # detection-only: returns (horizontal_list, free_list)
        # horizontal_list entries: [x_min, x_max, y_min, y_max]
        horizontal, _ = reader.detect(frame)
        for sublist in horizontal:       # detect() returns list-of-sublists
            for box in sublist:          # each box: [x_min, x_max, y_min, y_max]
                y_centers.append((box[2] + box[3]) / 2.0)

    cap.release()

    if len(y_centers) < 5:
        return None  # too few detections to cluster reliably

    ys = np.array(y_centers)
    top_frac = (ys < h * 0.30).sum() / len(ys)
    bot_frac = (ys > h * 0.70).sum() / len(ys)

    if top_frac >= 0.60:
        # ponytail: fixed 30% bands; add padding tuning if specific videos clip text
        return (0, 0, w, int(h * 0.30))
    if bot_frac >= 0.60:
        return (0, int(h * 0.70), w, h)

    return None  # scattered — full-frame scan
