import cv2
import numpy as np
import os
import re

jpg_pattern = re.compile('\w*\.jpg')
number_pattern = re.compile('\w*\.jpg')

basedir = 'PD8-files/car1'
txt_filename = 'PD8-files/gtcar1.txt'

all_files = [ os.path.join(basedir,i) for i in os.listdir(basedir) if jpg_pattern.search(i) is not None]

file_obj  = open(txt_filename, 'r')

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[5]

tracker = cv2.Tracker_create(tracker_type)

for idx,filename in enumerate(all_files):

    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    line = file_obj.readline()
    split = line.split(",")

    if 'NaN' not in split:

        x0 = int(float(split[0]))
        y0 = int(float(split[1]))
        x1 = int(float(split[2]))
        y1 = int(float(split[3]))

        img = cv2.imread(filename, cv2.IMREAD_COLOR)

        if idx == 0:
            ok = tracker.init(img, (x0,y0,x1-x0,y1-y0) )
        else:
            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(img)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(img, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)        


        cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),2)
    
    cv2.imshow('teste',img)

    a = cv2.waitKey(20) & 0xFF
    if a == ord('q'):
        break

file_obj.close()        