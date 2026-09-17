# Visual-Movement-Detection

A real-time research tool for detecting human body movement with a webcam and comparing it with live dual-channel EMG and joystick data from an ESP32 (or compatible serial device).

The application uses MediaPipe Pose landmarks to classify body regions as **moving**, **stationary**, or **uncertain**. It can run as a camera-only movement detector or alongside real EMG hardware. Motion and EMG are timestamped on the same PC clock so recorded sessions can be analysed together afterward.

## Features

- Real-time pose detection with OpenCV and MediaPipe
- Regional movement detection for head, eyes, arms, torso, and legs
- Landmark smoothing, body-size normalization, and hysteresis to reduce false movement detections
- Tracking for up to two people at once
- Camera-only and camera + EMG operating modes
- Real serial communication with ESP32 EMG hardware, including COM3, COM4, or any detected serial port
- Clear on-screen hardware states: `COM NOT CONNECTED`, `WAITING FOR DATA`, `RECEIVING DATA`, and `DATA STOPPED`
- Live raw EMG channel and joystick plots generated only from valid serial data
- Movement, EMG RMS, and CSV recording support
- Calibration trial labels for collecting research data

## Requirements

- Python 3.10+
- A webcam
- For EMG mode: an ESP32 or compatible device connected by USB serial

Install dependencies:

```bash
pip install opencv-python mediapipe numpy pyserial
```

## Model file

Place a compatible MediaPipe Pose Landmarker model at:

```text
models/pose_landmarker_full.task
```

The program expects this exact relative path beside `main.py`.

## Run

```bash
python main.py
```

At startup, choose either:

- **Visual Motion Only** — webcam movement detection without EMG hardware
- **Visual Motion + EMG Hardware** — webcam movement detection plus live serial EMG and joystick data

For EMG mode, select or enter the serial port (for example `COM3` or `COM4`) and the device baud rate. Leave the port blank to scan detected serial ports automatically.

## ESP32 serial format

Each device message must be a newline-terminated CSV row with 25 values:

```text
seq,timestamp_ms,emg1_0,emg1_1,...,emg1_9,emg2_0,emg2_1,...,emg2_9,joy_x,joy_y,btn
```

Example:

```text
1,1000,510,512,509,515,508,511,514,510,513,509,498,500,497,502,499,501,503,498,500,497,2048,2048,0
```

Only rows matching this format are accepted. Invalid or incomplete serial data is ignored, so it cannot create a false EMG trace.

## Controls

| Key | Action |
| --- | --- |
| `R` | Start/stop movement CSV recording |
| `L` | Start/stop optional EMG and joystick CSV logging (EMG mode) |
| `Tab` | Select the next tracked person for trial labeling |
| `0` | No trial |
| `1` | Still |
| `2` | Right arm |
| `3` | Left arm |
| `4` | Head |
| `5` | Right leg |
| `6` | Left leg |
| `7` | Whole body |
| `8` | Sitting still |
| `9` | Lying still |
| `S` | Lying sideways still |
| `Q` or `Esc` | Exit |

## Output data

CSV files are created only after recording starts and are saved in the `data/` folder beside the program.

- `movement_data_*.csv`
- contains frame-by-frame movement measurements, regional states, trial labels, and track IDs.
- `emg_data_*.csv`
- contains raw dual-channel EMG samples and joystick values received from the serial device.
