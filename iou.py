# Takes results tensor
# Pairs up bounding boxes that are locationally similar
# Thresholds low confidence boxes
    # if confidence < 40% remove
# Calculate IOU for pairs
    # if IOU > 0.8: 
        # take higher confidence to be prediction
    # if IOU < 0.8:
        # DEFECTIVE

# returns boolean (Defective)

# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2
import math

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def calc_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def is_defective(results):
    r = results.xywh[0]
    num_boxes = len(r)
    pair_tracker = []
    shit_index = []
    for i in range(r):
        if r[i][4] > 0.4: 
            dist = calc_dist(r[i])

            




