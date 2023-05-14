import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') 

def generate_boxes(camera_frame):
    frame = np.copy(camera_frame)
    preds=[]
    # Greyscale webcam input
    # Localize face with Haarcascade
    results = model(camera_frame) 
    if type(results!= None):
        for item in results.xywh[0]:
            item = item.numpy()
            bottomL = (item[0][0] - item[2][0]/2, item[1][0] - item[3][0]/2)
            topR = (item[0][0] + item[2][0]/2, item[1][0] + item[3][0]/2)
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