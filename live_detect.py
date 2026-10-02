# Example run:
#conda activate yolo-env1  
# python live_detect.py --model my_model.pt --source 0 --conf 0.4 --resolution 640x480 --output output.avi
#python live_detect.py --model yolo26n-pose.pt --source 0 --conf 0.4 --resolution 640x480 --output output.avi 
#
# This script is meant for live input such as a webcam or camera feed.
# It opens a stream, runs YOLO on each frame, shows the annotated image, and optionally saves it to a video file.

import argparse
import os

import cv2
from ultralytics import YOLO


def get_unique_output_path(output_path):
    # If the requested file already exists, create a new name such as output_1.avi, output_2.avi, etc.
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
    # Parse command-line arguments so the user can control the model, camera, confidence, and output file.
    parser = argparse.ArgumentParser(description="Live YOLO detection from a camera or webcam.")
    parser.add_argument("--model", required=True, help="Path to the YOLO weights file, e.g. my_model.pt")
    parser.add_argument("--source", default="0", help="Camera index or video path. Default: 0")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--resolution", default=None, help="Optional display size, example: 640x480")
    parser.add_argument("--output", default=None, help="Optional output video path, example: output.avi")
    return parser.parse_args()


def main():
    # Load the trained YOLO model from disk.
    args = parse_args()
    model = YOLO(args.model)

    # If the source is numeric, treat it as a webcam/camera index.
    # Otherwise, treat it as a file path (useful if you want to stream a saved video through this script too).
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    # If the camera/video could not be opened, stop with a helpful message.
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    writer = None
    if args.output:
        args.output = get_unique_output_path(args.output)
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # Keep reading frames until the stream ends or the user presses q.
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # If the user supplied a display size, resize the frame before inference.
        if args.resolution:
            width, height = map(int, args.resolution.split("x"))
            frame = cv2.resize(frame, (width, height))

        # Create the video writer once, the first time we need to save output.
        if writer is None and args.output:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                args.output,
                cv2.VideoWriter_fourcc(*"MJPG"),
                30,
                (width, height),
            )

        # Run detection on the current frame and draw boxes.
        results = model(frame, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        # Save the annotated frame if recording is enabled.
        if writer is not None:
            writer.write(annotated)

        # Show the live window to the user.
        cv2.imshow("YOLO Live Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Always clean up the camera and video writer before exiting.
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
