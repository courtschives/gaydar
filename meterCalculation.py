# Sapphic Visual Signal meter: turns per-frame YOLO detections into a 0-100 score
# and draws it in the bottom-left corner of the frame.

import json
import os

import cv2

WEIGHT_CLASS_POINTS = {
    "regular": 30,
    "high": 40,
    "super": 50,
}

METER_MAX = 100
DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weightClass.json")


def load_label_weights(json_path=DEFAULT_WEIGHTS_PATH):
    # Map each label to its weight class (regular/high/super) from weightClass.json.
    with open(json_path, "r") as f:
        entries = json.load(f)
    return {entry["label"]: entry["weightClass"] for entry in entries}


def compute_frame_score(result, label_weights):
    # Sum weight_points * confidence across every detection in the frame, capped at METER_MAX.
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0.0

    total = 0.0
    for box in boxes:
        cls_name = result.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        weight_class = label_weights.get(cls_name, "regular")
        points = WEIGHT_CLASS_POINTS.get(weight_class, WEIGHT_CLASS_POINTS["regular"])
        total += points * confidence

    return min(total, METER_MAX)


def meter_color(score):
    # Green -> amber -> red as the signal climbs, for a quick visual read (BGR for OpenCV).
    if score < 25:
        return (0, 0, 255)
    elif score < 75:
        return (0, 200, 255)
    return (0, 200, 0)


def draw_meter(frame, score, label="Sapphic Visual Signal"):
    # Draw the meter bar, label, and percentage in the bottom-left corner of the frame.
    height = frame.shape[0]
    percentage = int(round(score))

    margin = 20
    bar_width = 220
    bar_height = 22
    bar_x = margin
    bar_y = height - margin - bar_height

    cv2.putText(
        frame, label, (bar_x, bar_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

    fill_width = int(bar_width * (score / METER_MAX))
    if fill_width > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), meter_color(score), -1)

    cv2.putText(
        frame, f"{percentage}%", (bar_x + bar_width + 10, bar_y + bar_height - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )

    return frame
