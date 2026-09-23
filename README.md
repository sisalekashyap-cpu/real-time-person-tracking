# AI-Powered Person Tracking & Attendance System (Corrected Version)

A real-time Computer Vision system for detecting, tracking, and counting people. Designed for classrooms, university halls, and corporate workspaces.

This version is updated with:
- **Live Real Camera / Webcam support** as default (`/dev/video0` or any connected camera).
- **Fast, native OpenCV C++ Non-Maximum Suppression (`cv2.dnn.NMSBoxes`)**.
- **Corrected Centroid Tracking Logic** (fixed dropped detections and un-deregistered objects).
- **Accurate Real-Time FPS** and clear Heads-Up Display (HUD).
- **Proper MobileNet-SSD 300×300 input blob geometry**.
- **Modern dependency support** in `requirements.txt`.
- **Command-line arguments (`argparse`)** to easily switch cameras, videos, and parameters.

---

## 1. Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run with Real Camera (Default)

To run using your computer's default webcam:

```bash
python person_tracking.py
```
*(Automatically connects to camera index `0`)*

If you have multiple webcams (e.g., external USB camera on index `1`):

```bash
python person_tracking.py --source 1
```

### Step 3: Run with Sample Video

To test with a recorded video file:

```bash
python person_tracking.py --source test_video.mp4
```

To loop the video continuously:

```bash
python person_tracking.py --source test_video.mp4 --loop
```

To stop the application at any time, press **`q`** or **`ESC`** in the video window.

---

## 2. Command-Line Options

| Option | Flag | Default | Description |
|---|---|---|---|
| `--source` | `-s` | `0` | Camera device index (`0`, `1`) or path to a video file. |
| `--confidence` | `-c` | `0.5` | Minimum confidence score to filter person detections (`0.0`–`1.0`). |
| `--nms-thresh` | `-n` | `0.3` | Non-Maximum Suppression overlap threshold. |
| `--max-disappeared` | `-d` | `50` | Frames an object can be missing before being deregistered. |
| `--max-distance` | `-m` | `90` | Maximum pixel distance to match an object's centroid. |
| `--width` | | `640` | Frame display and processing width (in pixels). |
| `--loop` | | `False` | Loop video playback when EOF is reached (video files only). |

### Examples

**Higher sensitivity in low-light environments:**
```bash
python person_tracking.py --source 0 --confidence 0.35
```

**Fast movement / large room (allow wider distance jumps):**
```bash
python person_tracking.py --source 0 --max-distance 130
```

---

## 3. What Was Fixed From The Original Version

1. **Live Camera Default**:
   - Original only opened `test_video.mp4`.
   - Now defaults to live webcam input with camera index auto-parsing, camera buffer latency reduction, and graceful fallback.

2. **Centroid Tracking Logic Defect Fixed**:
   - The original code used an exclusive `if D.shape[0] >= D.shape[1]: ... else: ...` block that discarded newly appeared people or failed to deregister disappeared objects when distances exceeded `maxDistance`.
   - The corrected tracker decouples `unusedRows` (disappeared) and `unusedCols` (new registrations) so both are handled independently.

3. **OpenCV Native C++ NMS**:
   - Replaced the 40-line custom NumPy NMS (which failed when no detections were present or returned `None` on errors) with OpenCV's optimized `cv2.dnn.NMSBoxes`.

4. **Correct MobileNet-SSD Input Resolution**:
   - Replaced arbitrary frame size passing with MobileNet-SSD's native `(300, 300)` input blob geometry, preventing distorted aspect ratios.

5. **Accurate FPS Calculation**:
   - Replaced integer-truncated `time_diff.seconds` (which showed `FPS: 0.00` for the first second) with a high-resolution rolling window counter using `time.perf_counter()`.

6. **Clear Statistics HUD**:
   - Separated **Active Count** (people currently visible in frame) from **Total Unique Seen** (cumulative tracking IDs assigned).
   - Ensured bounding box labels never clip off-screen.

7. **Clean Exit and Resource Management**:
   - All streams and windows are released cleanly in a `try...finally` block.
