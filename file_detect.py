# Example run:
#conda activate yolo-env1  
# python file_detect.py --model my_model.pt --source "C:/videos/test.mp4" --conf 0.4 --resolution 720x1280 --output output.avi
# python file_detect.py --model my_model.pt --source "C:/images_folder" --conf 0.4
# python file_detect.py --model my_model.pt --source "testVideo/originals/output6.avi" --conf 0.5 --resolution 720x1280 --output testVideo/testOutput6.avi
#
# This script is meant for saved inputs: a single image, a folder of images/videos, or a video file.
# It checks the input type, runs YOLO on each frame or image, displays the annotated output, and optionally writes it out.

import argparse
import csv
import glob
import os

import cv2
from ultralytics import YOLO

from meterCalculation import compute_frame_score, draw_meter, load_label_weights


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VALID_VIDEO_EXTS = {".avi", ".mov", ".mp4", ".mkv", ".wmv"}


def get_unique_output_path(output_path):
    # If the requested file already exists, create a new name like output_1.avi, output_2.avi, and so on.
    if not output_path or not os.path.exists(output_path):
        return output_path

    base, ext = os.path.splitext(output_path)
    counter = 1
    new_path = output_path
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_path


def parse_args():
    # Parse command-line arguments so the user can control the model, input file, confidence, and output.
    parser = argparse.ArgumentParser(description="YOLO detection for saved images, image folders, or videos.")
    parser.add_argument("--model", required=True, help="Path to the YOLO weights file, e.g. my_model.pt")
    parser.add_argument("--source", required=True, help="Image, video file, or folder path")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--resolution", default=None, help="Optional resize size for display, example: 640x480")
    parser.add_argument("--output", default=None, help="Optional output path for processed media")
    return parser.parse_args()


def resize_frame(frame, res):
    # If no resize is requested, return the original frame.
    if res is None:
        return frame
    width, height = map(int, res.split("x"))
    return cv2.resize(frame, (width, height))


def make_writer(output_path, width, height, fps):
    # Create the output directory if needed so the script can write the saved result.
    # XVID is used instead of MJPG: on Windows, OpenCV's MJPG/AVI writer frequently corrupts
    # color and padding for arbitrary resolutions, producing a green cast and skewed aspect ratio.
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))


def get_source_fps(cap, default=30.0):
    # Some containers/codecs report an invalid fps (0 or NaN); fall back to a sane default.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        return default
    return fps


def get_detections_csv_path(video_path, output_path):
    # Pair the detections log with the output video name when one is given,
    # otherwise base it on the source video name so it's easy to match up in TouchDesigner.
    base, _ = os.path.splitext(output_path or video_path)
    return base + "_detections.csv"


def make_detections_writer(csv_path):
    # Open a CSV file for per-frame detection logging and write the header row.
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "class", "confidence", "x1", "y1", "x2", "y2", "meter_score"])
    return csv_file, writer


def log_detections(writer, frame_index, result, meter_score):
    # Write one row per detected box (frame, class, confidence, bounding box corners, meter score).
    # Frames with no detections still get one row so the meter timeline has no gaps for TouchDesigner.
    boxes = result.boxes
    score_str = f"{meter_score:.2f}"

    if boxes is None or len(boxes) == 0:
        writer.writerow([frame_index, "", "", "", "", "", "", score_str])
        return

    for box in boxes:
        cls_name = result.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        writer.writerow([frame_index, cls_name, f"{confidence:.4f}", f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", score_str])


def process_image(model, image_path, conf, resolution=None, output_path=None):
    # Read the image, resize it if needed, and run inference.
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    frame = resize_frame(frame, resolution)
    results = model(frame, conf=conf, verbose=False)
    annotated = results[0].plot()

    # If an output path is given, save the annotated image.
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, annotated)

    cv2.imshow("YOLO File Detection", annotated)
    cv2.waitKey(0)


def process_video(model, video_path, conf, resolution=None, output_path=None):
    # Open the video and process each frame in order.
    label_weights = load_label_weights()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    if output_path:
        output_path = get_unique_output_path(output_path)

    fps = get_source_fps(cap)
    csv_path = get_detections_csv_path(video_path, output_path)
    csv_file, detections_writer = make_detections_writer(csv_path)

    writer = None
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_frame(frame, resolution)
            results = model(frame, conf=conf, verbose=False)
            annotated = results[0].plot()
            score = compute_frame_score(results[0], label_weights)
            draw_meter(annotated, score)
            log_detections(detections_writer, frame_index, results[0], score)
            frame_index += 1

            if writer is None and output_path:
                height, width = annotated.shape[:2]
                writer = make_writer(output_path, width, height, fps)

            if writer is not None:
                writer.write(annotated)

            cv2.imshow("YOLO Video Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

    print(f"Detections log saved to: {csv_path}")


def process_folder(model, folder_path, conf, resolution=None, output_dir=None):
    # Gather all supported image/video files from the folder.
    label_weights = load_label_weights()
    file_paths = []
    for file in sorted(glob.glob(os.path.join(folder_path, "*"))):
        ext = os.path.splitext(file)[1].lower()
        if ext in VALID_IMAGE_EXTS or ext in VALID_VIDEO_EXTS:
            file_paths.append(file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process each file in the folder one by one.
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()

        if ext in VALID_IMAGE_EXTS:
            frame = cv2.imread(file_path)
            if frame is None:
                continue
            frame = resize_frame(frame, resolution)
            results = model(frame, conf=conf, verbose=False)
            annotated = results[0].plot()

            if output_dir:
                output_path = os.path.join(output_dir, os.path.basename(file_path))
                cv2.imwrite(output_path, annotated)

            cv2.imshow("YOLO Folder Detection", annotated)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

        elif ext in VALID_VIDEO_EXTS:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                continue

            if output_dir:
                out_name = os.path.splitext(os.path.basename(file_path))[0] + ".avi"
                video_output_path = os.path.join(output_dir, out_name)
            else:
                video_output_path = None

            fps = get_source_fps(cap)
            csv_path = get_detections_csv_path(file_path, video_output_path)
            csv_file, detections_writer = make_detections_writer(csv_path)

            writer = None
            frame_index = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    frame = resize_frame(frame, resolution)
                    results = model(frame, conf=conf, verbose=False)
                    annotated = results[0].plot()
                    score = compute_frame_score(results[0], label_weights)
                    draw_meter(annotated, score)
                    log_detections(detections_writer, frame_index, results[0], score)
                    frame_index += 1

                    if writer is None and video_output_path:
                        height, width = annotated.shape[:2]
                        writer = make_writer(video_output_path, width, height, fps)

                    if writer is not None:
                        writer.write(annotated)

                    cv2.imshow("YOLO Folder Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            finally:
                cap.release()
                if writer is not None:
                    writer.release()
                csv_file.close()

            print(f"Detections log saved to: {csv_path}")

    cv2.destroyAllWindows()


def main():
    args = parse_args()
    model = YOLO(args.model)

    source = args.source
    ext = os.path.splitext(source)[1].lower()

    if args.output:
        args.output = get_unique_output_path(args.output)

    # If the source is a folder, process all valid files in it.
    if os.path.isdir(source):
        process_folder(model, source, args.conf, args.resolution, args.output)
        return

    # Single image input.
    if ext in VALID_IMAGE_EXTS:
        process_image(model, source, args.conf, args.resolution, args.output)
        return

    # Single video input.
    if ext in VALID_VIDEO_EXTS:
        process_video(model, source, args.conf, args.resolution, args.output)
        return

    raise ValueError(f"Unsupported file type: {source}")


if __name__ == "__main__":
    main()
