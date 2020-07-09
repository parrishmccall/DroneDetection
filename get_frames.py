import cv2
import numpy as np
from time import sleep
import os

video_capture = cv2.VideoCapture("*")
video_capture.set(cv2.CAP_PROP_FPS, 30)

count = 0
frames = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
    else:
        sleep(0.0000001)
        ret, frame = video_capture.read()  # Grabs, decodes and returns the next video frame (Capture frame-by-frame)

        cv2.imwrite('frames/' + 'drone1'+ str(count) + '.jpg', frame)
        count += 1

        frame = cv2.resize(frame, (800, 500))

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
