import torch
import cv2
import cvzone
import numpy as np
import math
from sort import *

className=["White Spot"]

model = torch.hub.load("ultralytics/yolov5", "custom", "best4.pt")
cap = cv2.VideoCapture("videos/vid.mp4")

tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)           #Tracking the objects through multiple frames

#Limits for the lines
limits1=[5,300,1275,300]
limits2=[5,350,1275,350]
# limits3=[5,355,1275,355]

counts=[]           #List of object IDS
finalCount=[]       #Final number of objects counted

while True:
    success,img =cap.read()
    result=model(img)

    detections = result.pred[0].numpy()

    resultsTracker = tracker.update(detections)

    line1 = cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 1)
    line2 = cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 1)
    cvzone.putTextRect(img, f"Count : {finalCount}", (50, 50))

    for result in resultsTracker.tolist():
        x1,y1,x2,y2,id=int(result[0]),int(result[1]),int(result[2]),int(result[3]),int(result[4])
        w, h = x2 - x1, y2 - y1


        cvzone.putTextRect(img, f"{id}", (max(0, x1), max(35, y1)), thickness=1, scale=1.5,
                           offset=1)

        cvzone.cornerRect(img, (x1, y1, w, h), l=1, rt=1, colorR=(0, 0, 0))

        # line3 = cv2.line(img, (limits3[0], limits3[1]), (limits3[2], limits3[3]), (0, 0, 255), 1)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

        if limits1[0] < cx < limits1[2] and limits1[1] - 5 < cy < limits1[1] + 5:
            if counts.count(id) == 0:
                counts.append(id)
                line1 = cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 3)

        if limits2[0] < cx < limits2[2] and limits2[1] - 5 < cy < limits2[1] + 5:
            if counts.count(id) == 0:
                counts.append(id)
                line2 = cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 3)

        finalCount=len(counts)

    cv2.imshow("Image",img)

    key=cv2.waitKey(1)
    if key==27:
        break

print(finalCount)
cap.release()


