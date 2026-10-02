# gaydar

A custom YOLO object detector designed to identify sapphic visual signals. Each frame's detections are combined into a 0–100 **Sapphic Visual Signal** meter that's drawn on the video, and every detection is logged to a CSV that TouchDesigner (or anything else) can read.

## Example output

[`examples/testOutput4.avi`](examples/testOutput4.avi) shows annotated output: boxes and labels on each detection, with the meter in the bottom-left corner. [`examples/testOutput4_detections.csv`](examples/testOutput4_detections.csv) is the detection log for the same clip. GitHub can't play `.avi` in the browser, so download the file to watch it.

## What's in here

| File | What it does |
| --- | --- |
| `my_model.pt` | The trained model (YOLO26-L base, 18 classes, 640px input). |
| `file_detect.py` | Runs the model on an image, a video or a folder of either. For video it draws the meter, saves an annotated `.avi` and writes a `<output>_detections.csv`. |
| `live_detect.py` | Runs the model on a webcam or live stream and can record the result (boxes only, no meter). |
| `yolo_detect.py` | The original general-purpose detection script from EJ Technology Consultants. It supports USB cameras and Raspberry Pi cameras and shows an FPS counter. |
| `meterCalculation.py` | Turns a frame's detections into the meter score and draws the meter. |
| `weightClass.json` | Maps each label to a weight class (`regular`, `high` or `super`). |
| `gaydar.py` | The training script, exported from Google Colab: k-fold validation followed by a final full-dataset training run. |
| `train/` | `args.yaml` (training settings) and `results.csv` (per-epoch metrics) from the final training run. |
| `Train_YOLO_Models.ipynb` | Earlier Colab notebook this project started from. |

### Classes

`baggyAssPant`, `bigAssShorts`, `carabiner`, `collarOrButtons`, `coloredHair`, `drasticMakeup`, `funkyBrows`, `gayAssHat`, `gayAssPiercing`, `gayAssStreetwear`, `hightop`, `largeJewelry`, `lineUp`, `locs`, `mulletsNFlow`, `muscleTee`, `vest`, `yourBoxersAreShowing`

### How the meter works

For every detection in a frame:

```
points = {regular: 30, high: 40, super: 50}[weightClass] × confidence
```

The points for all detections are added up and capped at 100. The bar is red below 25, amber from 25 to 74 and green from 75 up. Labels missing from `weightClass.json` (currently `coloredHair` and `gayAssStreetwear`) count as `regular`. To re-weight the meter, edit `weightClass.json`; no retraining is needed.

### Detection CSV format

```
frame,class,confidence,x1,y1,x2,y2,meter_score
```

The CSV has one row per detection. A frame with no detections still gets one row with an empty class, so the meter timeline has no gaps.

## Run it locally

1. **Install Python 3.10+** and, ideally, create a virtual environment (venv or conda).
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   For GPU inference, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first. The model also runs on CPU, just more slowly.
3. **Run it on a video:**
   ```bash
   python file_detect.py --model my_model.pt --source path/to/video.mp4 --conf 0.4 --output out/result.avi
   ```
   `--resolution 720x1280` resizes frames before inference (format: width x height). If the output file already exists, the script adds a number to the new file's name instead of overwriting it.
4. **Run it on a webcam:**
   ```bash
   python live_detect.py --model my_model.pt --source 0 --conf 0.4 --output live.avi
   ```

Press `q` to close the preview window. Keep `meterCalculation.py` and `weightClass.json` in the same folder as `file_detect.py`.

## Train your own

The training data isn't included in this repo. To train your own model:

1. **Collect images** of the styles or objects you want to detect. Each class needs at least a few dozen examples, and more is better.
2. **Label them** in YOLO format with a tool such as [Label Studio](https://labelstud.io/). Export the labels and organise them like this:
   ```
   data.zip
   ├── images/        # .jpg / .png
   ├── labels/        # one .txt per image, same file name
   └── classes.txt    # one class name per line
   ```
3. **Open `gaydar.py` in Google Colab** with a GPU runtime. It's a Colab export, so lines starting with `!` are notebook shell commands and the script won't run as plain Python. Set `BASE` to wherever `data.zip` lives in your Google Drive, then run the cells in order:
   - **K-fold training** splits the dataset 5 ways and trains `yolo26s.pt` for 60 epochs on each split. It reports mAP50 for each fold and the average. This shows how well the model generalises, which matters with a small, imbalanced dataset.
   - **Final model** trains `yolo26l.pt` for 120 epochs on the whole dataset. Because the validation set is the training set, the metrics from this run are only a sanity check.
4. **Export the model:** copy `final_model/train/weights/best.pt` and rename it `my_model.pt`.
5. **Update `weightClass.json`** so it lists your class names and their weights.

## Credits

The training notebook and `yolo_detect.py` are adapted from Evan Juras / [EJ Technology Consultants](https://ejtech.io): [Train-and-Deploy-YOLO-Models](https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models). Detection uses [Ultralytics YOLO](https://docs.ultralytics.com/) (AGPL-3.0).
