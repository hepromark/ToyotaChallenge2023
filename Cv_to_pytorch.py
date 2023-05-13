import cv2

import numpy as np
import torch

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, dsize=(640, 640))
    print(frame)
    CHW = np.transpose(frame, (2, 0, 1))
    # blob = cv2.dnn.blobFromImage(frame, 1, size=size,swapRB=True)
    cv2.imshow('frame', frame)
    print(CHW.shape)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()


cv2.destroyAllWindows()


