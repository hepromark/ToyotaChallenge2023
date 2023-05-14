import cv2

import numpy as np
import torch

cap = cv2.VideoCapture(0)

# size = (640,640)
while True:
    ret, frame = cap.read()

    #set height and width
    frame = cv2.resize(frame, dsize=(640, 640))
    
    #changing to CHW
    CHW = np.transpose(frame, (2, 0, 1))

    cv2.imshow('frame', frame)
    print(CHW.shape)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()




