import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import os
# Default called trackbar function
def setValues(x):
    print("")
# Function to identify the shape based on contour matchingQ
def identify_shape(cnt):
    epsilon = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Calculate solidity (ratio of contour area to its convex hull area)
    hull = cv2.convexHull(cnt)
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area

    # Calculate aspect ratio of the bounding rectangle
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h

    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif len(approx) >= 9 and solidity > 0.7:
        return "Circle"
    else:
        return "Unknown"
# Creating the trackbars needed for adjusting the marker color
cv2.namedWindow("Color detectors")
cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180, setValues)
cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, setValues)
cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255, setValues)
cv2.createTrackbar("Lower Value", "Color detectors", 49, 255, setValues)

# Giving different arrays to handle color points of different color
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific color
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Here is code for Canvas setup
paintWindow = np.zeros((520, 720, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

# Add two buttons for Save and Reopen
paintWindow=cv2.rectangle(paintWindow, (610, 1), (700, 33), (0, 0, 0), 2)
cv2.putText(paintWindow, "SAVE", (612, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
#paintWindow=cv2.rectangle(paintWindow, (610, 40), (700, 71), (0, 0, 0), 2)
#cv2.putText(paintWindow, "REOPEN", (612, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Loading the default webcam of PC.
cap = cv2.VideoCapture(0)

# Variable to track drawing state
drawing = False

# Variable to store the current canvas image
canvas_image = np.copy(paintWindow)

# Function to save the canvas as a .png file
def save_canvas():
    cv2.imwrite("canvas_image.png", paintWindow[67:, :, :])

# Function to reopen a saved .png file
def reopen_canvas():
    global canvas_image
    if os.path.exists("canvas_image.png"):
        canvas_image = cv2.imread("canvas_image.png")
        paintWindow[67:, :, :] = 255  # Clear canvas
        paintWindow[67:, :, :] = canvas_image


while True:
    # ... (Previous code remains unchanged)
    # Reading the frame from the camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Hand detection using MediaPipe
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extracting index finger and thumb coordinates
            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            height, width, _ = frame.shape
            index_finger_x, index_finger_y = int(index_finger_landmark.x * width), int(index_finger_landmark.y * height)
            thumb_x, thumb_y = int(thumb_landmark.x * width), int(thumb_landmark.y * height)

            # Draw a circle at the tip of the index finger
            cv2.circle(frame, (index_finger_x, index_finger_y), 15, (255, 255, 255), -1)

            # Check for pinch gesture (distance between thumb and index finger)
            distance = np.sqrt((thumb_x - index_finger_x) ** 2 + (thumb_y - index_finger_y) ** 2)

            # If pinch in occurs, start drawing
            if not drawing and distance < 30:
                drawing = True

            # If pinch out occurs, stop drawing and start a new drawing selection
            elif drawing and distance >= 30:
                drawing = False
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

            if drawing:
                mask = np.zeros_like(frame)
                cv2.circle(mask, (index_finger_x, index_finger_y), 15, (255, 255, 255), -1)

                index_finger_masked = cv2.bitwise_and(frame, mask)

                gray_frame = cv2.cvtColor(index_finger_masked, cv2.COLOR_BGR2GRAY)

                _, thresholded = cv2.threshold(gray_frame, 30, 255, cv2.THRESH_BINARY)

                cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                center = None

                if len(cnts) > 0:
                    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
                    identified_shape = identify_shape(cnt)

                    # Replace the drawn shape with a perfect shape
                    if identified_shape != "Unknown":
                        perfect_shape = replace_with_perfect_shape(cnt, identified_shape)
                        cv2.drawContours(frame, [perfect_shape], 0, (255, 255, 255), -1)  # Replace with white circle
                    else:
                        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

                    M = cv2.moments(cnt)
                    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

                    # ... (Previous code remains unchanged)
                    
                    if center[1] <= 65:
                        if 40 <= center[0] <= 140:  # Clear Button
                            bpoints = [deque(maxlen=512)]
                            gpoints = [deque(maxlen=512)]
                            rpoints = [deque(maxlen=512)]
                            ypoints = [deque(maxlen=512)]

                            blue_index = 0
                            green_index = 0
                            red_index = 0
                            yellow_index = 0

                            paintWindow[67:, :, :] = 255
                            drawing = False
                        elif 160 <= center[0] <= 255:
                            colorIndex = 0  # Blue
                        elif 275 <= center[0] <= 370:
                            colorIndex = 1  # Green
                        elif 390 <= center[0] <= 485:
                            colorIndex = 2  # Red
                        elif 505 <= center[0] <= 600:
                            colorIndex = 3  # Yellow
                        elif 610 <= center[0] <= 636:
                            if 1 <= center[1] <= 33:  # SAVE button
                                save_canvas()
                            elif 40 <= center[1] <= 71:  # REOPEN button
                                reopen_canvas()
                    else:
                        if colorIndex == 0:
                            bpoints[blue_index].appendleft(center)
                        elif colorIndex == 1:
                            gpoints[green_index].appendleft(center)
                        elif colorIndex == 2:
                            rpoints[red_index].appendleft(center)
                        elif colorIndex == 3:
                            ypoints[yellow_index].appendleft(center)
            
            # Draw lines of all the colors on the canvas and frame
            points = [bpoints, gpoints, rpoints, ypoints]
            for i in range(len(points)):
                for j in range(len(points[i])):
                    # Use interpolation to connect consecutive points with smooth curves
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                        cv2.circle(frame, points[i][j][k], 8, colors[i], -1)    

# ... (Rest of the code remains unchanged)
    # Show all the windows

    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    # If the 'q' key is pressed then stop the application
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()
