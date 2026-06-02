# Screenshots & Demo

This folder contains screenshots and demo images showcasing the Face AI System capabilities.

## Contents

### Existing Screenshots
- `img_*.jpg` - Screenshots captured during hand gesture detection (auto-captured when Open Palm gesture detected)

### Screenshot Categories

#### 1. Mask Detection Examples
Expected screenshots:
- `mask_detection_demo.jpg` - Real-time mask detection with confidence score
- `mask_multiple_faces.jpg` - Multiple faces detected with mask status
- `no_mask_detection.jpg` - NO MASK detection example

#### 2. Emotion Detection Examples
Expected screenshots:
- `emotion_happy.jpg` - Happy emotion detection
- `emotion_sad.jpg` - Sad emotion detection
- `emotion_surprise.jpg` - Surprise emotion detection
- `emotion_angry.jpg` - Angry emotion detection
- `emotion_neutral.jpg` - Neutral emotion detection

#### 3. GUI Screenshots
Expected screenshots:
- `gui_main.jpg` - Main control panel interface
- `gui_active_detection.jpg` - GUI during active detection
- `gui_statistics.jpg` - Statistics panel view

#### 4. Hand Gesture Examples
Existing:
- `img_*.jpg` - Auto-captured screenshots from hand gesture detection

Expected additional:
- `hand_peace.jpg` - Peace sign detection
- `hand_thumbs_up.jpg` - Thumbs up detection
- `hand_fist.jpg` - Fist detection
- `hand_open_palm.jpg` - Open palm detection

#### 5. Object Detection Examples
Expected screenshots:
- `yolo_detection.jpg` - YOLO object detection example
- `yolo_multiple_objects.jpg` - Multiple objects detected

## Capturing New Screenshots

### Manual Capture
Press `Print Screen` or use snipping tool during detection.

### Auto-Capture (Hand Gesture)
Make an **Open Palm** gesture during hand gesture detection. Images are automatically saved to this folder every 2 seconds.

### Programmatic Capture
```python
import cv2
# In detection loop:
if ret:
    cv2.imwrite("assets/screenshots/my_screenshot.jpg", frame)
```

## Demo Video/GIF

Consider adding:
- `demo_mask.gif` - Short GIF of mask detection
- `demo_emotion.gif` - Short GIF of emotion detection
- `demo_video.mp4` - Full demonstration video

## Tips for Good Screenshots

1. **Good lighting** - Ensure face is well-lit
2. **Front-facing camera** - Use webcam for best results
3. **Clear background** - Minimize clutter
4. **Close distance** - Face should fill 30-50% of frame
5. **Neutral expression** (for mask detection) - Smile naturally for emotion
