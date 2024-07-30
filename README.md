- Project Title :
AirCanvas

- Description :
A Python-based project that uses computer vision and MediaPipe to create a virtual canvas where users can draw in the air using hand gestures detected through a webcam.

- Features :
1. Hand gesture detection using MediaPipe
2. Drawing with multiple colors
3. Shape recognition (triangle, square, rectangle, circle)
4. Save and reopen canvas

- How It Works :
Explanation of key functionalities:
1. Hand Detection: Uses MediaPipe to detect hand landmarks.
2. Drawing: Tracks index finger position to draw on a virtual canvas.
3. Shape Recognition: Identifies shapes drawn on the canvas and replaces them with perfect shapes.
4. Save/Load: Save the current state of the canvas to a file and reopen it later.


## Installation
To install the necessary dependencies, run:
```sh
pip install numpy opencv-python mediapipe

