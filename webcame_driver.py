import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import math

# Helper functions
def calc_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def iou(center1,width1 , height1, center2, width2, height2):
    """
    Input: 
        center 1 - (x, y) center position of box 1
        width 1 - width of first box
        height 1 - height of first box
        center2 -- tuple representing the center point of the second box in the format (x2, y2)
        width2 -- width of the second box
        height2 -- height of the second box

        Outputs: IOU value of bounding boxes
    """
    # Calculate the coordinates of the first bounding box
    x1 = center1[0] - (width1 / 2)
    y1 = center1[1] - (height1 / 2)
    x2 = center1[0] + (width1 / 2)
    y2 = center1[1] + (height1 / 2)

    # Calculate the coordinates of the second bounding box
    x3 = center2[0] - (width2 / 2)
    y3 = center2[1] - (height2 / 2)
    x4 = center2[0] + (width2 / 2)
    y4 = center2[1] + (height2 / 2)

    # Calculate the coordinates of the intersection rectangle
    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)

     # Calculate the area of intersection rectangle
    intersection_area = max(0, x6 - x5) * max(0, y6 - y5)

    # Calculate the area of both bounding boxes
    box1_area = width1 * height1
    box2_area = width2 * height2

    # Calculate the IOU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

# Get results array
def result_parse(results):
    r = results.xywh[0].numpy()
    num_boxes = r.shape[0]
    pair_tracker = []
    bad_index = set()
    taken = set()
    error_index = set()
    outputs = [i for i in range(num_boxes)]

    # Looping through bounding boxes
    for i in range(num_boxes):
        if r[i][4] < 0.4:
            bad_index.add(i)
            continue

        if i in bad_index:
            continue

        dist_tracker = [30,0] # (distance, #index of second box that is close by)
        # Looping through remaining boxes
        for k in range(num_boxes):
            if r[k][4] < 0.4:
                bad_index.add(k)
                continue
            if k in bad_index:
                continue
            if i != k and r[i][5] != r[k][5]:
                # Get distance
                new_dist = calc_dist(r[i][0], r[i][1], r[k][0], r[k][1])
                if dist_tracker[0] > new_dist:
                    dist_tracker[0] = new_dist
                    dist_tracker[1] = k 
                    

        # Append pair of different labels to output list
        if dist_tracker[1] != 0 and i not in taken and k not in taken:
            pair_tracker.append([i, dist_tracker[1]])
            taken.add(i)
            taken.add(dist_tracker[1])

    # Check overlap
    for pairs in pair_tracker:
        x1, y1, w1, h1, _, _ = r[pairs[0]]
        x2, y2, w2, h2, _, _ = r[pairs[1]]
        
        if iou((x1, y1), w1, h1, (x2, y2), w2, h2) < 0.8:
            bad_index.add(pairs[0])
            error_index.add(pairs[1])
        else: # same overlap
            # take pair with higher prediction
            if r[pairs[0]][4] > r[pairs[1]][4]:
                bad_index.add(pairs[1])
            else:
                bad_index.add(pairs[0])

    # Making list output
    for i in range(len(outputs)):
        if outputs[i] in bad_index:
            outputs[i] = 'non'
        elif outputs[i] in error_index:
            
            outputs[i] = 'Defect'
        else:
            class_name = r[outputs[i]][5]
            
            if class_name == 1.0:
                outputs[i] = 'Sticker'
            else:
                outputs[i] = 'Hole'
    return outputs

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') 

# IOU
# import the necessary packages
# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def generate_boxes(camera_frame):
    frame = np.copy(camera_frame)
    preds=[]
    # Greyscale webcam input
    # Localize face with Haarcascade
    results = model(camera_frame) 
    print_val = result_parse(results)
    counter = 0

    colors  = {
        "Sticker": (255,0,0),#b
        "Hole": (0,255,0),#g
        "Defect": (0,0,255)#r
    }

    if type(results!= None) and type(print_val != None):
        for item in results.xywh[0]:
            if print_val[counter] == "non":
                counter +=1
                continue
            else:
                item = item.numpy()
                bottomL = (int(item[0] - item[2]/2), int(item[1] - item[3]/2))
                topR = (int(item[0] + item[2]/2), int(item[1] + item[3]/2))
                cv2.rectangle(frame, bottomL, topR, colors[print_val[counter]], 2 )
                #cv2.putText(image, text, position, font, fontsize, color=color)
                cv2.putText(frame, print_val[counter], bottomL, cv2.FONT_HERSHEY_PLAIN, 300, colors[print_val[counter]])
                counter +=1

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