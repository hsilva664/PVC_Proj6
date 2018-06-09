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


for filename in all_files:
    line = file_obj.readline()
    split = line.split(",")
    x0 = int(float(split[0]))
    y0 = int(float(split[1]))
    x1 = int(float(split[2]))
    y1 = int(float(split[3]))

    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),2)

    cv2.imshow('teste',img)

    a = cv2.waitKey(20) & 0xFF
    if a == ord('q'):
        break

file_obj.close()        