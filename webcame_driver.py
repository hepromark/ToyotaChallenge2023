import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple


# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') 

# IOU
# import the necessary packages
# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def generate_boxes(camera_frame):
    frame = np.copy(camera_frame)
    preds = False
    # Greyscale webcam input
    # Localize face with Haarcascade
    results = model(camera_frame) 
    if type(results!= None):
        for item in results.xywh[0]:
            item = item.numpy()
            print("_____________")
            print(type(item[0]))
            print(type(item[1]))
            print(type(item[2]))
            print(type(item[3]))
            print("_____________")
            bottomL = (int(item[0] - item[2]/2), int(item[1] - item[3]/2))
            topR = (int(item[0] + item[2]/2), int(item[1] + item[3]/2))
            cv2.rectangle(frame, bottomL, topR, (255,0,0), 2 )

    return frame, preds


camera=cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    if type(frame) != None:
        if frame.any() == True:
            frame, preds = generate_boxes(frame)
            cv2.imshow('Stickers or holes', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
camera.release()
cv2.destroyAllWindows()