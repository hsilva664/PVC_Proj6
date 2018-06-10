import cv2
import numpy as np
import os
import re

#RECTANGLE FUNCTIONS
#Inputs should be (x0,y0,x1,y1)
def union(a,b, imshape):
    imshape = (imshape[0]*2,imshape[1]*2)
    offset_i = (imshape[0]//4)
    offset_j = (imshape[1]//4)

    a = (a[0] + offset_j, a[1] + offset_i, a[2] + offset_j, a[3] + offset_i)
    b = (b[0] + offset_j, b[1] + offset_i, b[2] + offset_j, b[3] + offset_i)

    zeroed = np.zeros(imshape, dtype = np.bool)    
    zeroed[ a[1]:a[3], a[0]:a[2] ] = True
    zeroed[ b[1]:b[3], b[0]:b[2] ] = True


    return np.sum(zeroed)

def intersection(a,b, imshape):

    imshape = (imshape[0]*2,imshape[1]*2)
    offset_i = (imshape[0]//4)
    offset_j = (imshape[1]//4)

    a = (a[0] + offset_j, a[1] + offset_i, a[2] + offset_j, a[3] + offset_i)
    b = (b[0] + offset_j, b[1] + offset_i, b[2] + offset_j, b[3] + offset_i)

    zeroed_one = np.zeros(imshape, dtype = np.bool)    
    zeroed_one[ a[1]:a[3], a[0]:a[2] ] = True

    zeroed_two = np.zeros(imshape, dtype = np.bool)    
    zeroed_two[ b[1]:b[3], b[0]:b[2] ] = True

    result = np.logical_and(zeroed_one, zeroed_two)

    # v_a = np.zeros(imshape, dtype = np.bool)
    # v_a[ a[1]:a[3], a[0]:a[2] ] = True
    # cv2.imshow('a', 255*v_a.astype(np.uint8))
    # v_b = np.zeros(imshape, dtype = np.bool)
    # v_b[ b[1]:b[3], b[0]:b[2] ] = True
    # cv2.imshow('b', 255*v_b.astype(np.uint8))

    # cv2.imshow('zeroed', 255*result.astype(np.uint8))
    # cv2.waitKey(0)

    return np.sum(result)


a = (20,20,80,80)
b = (60,30,180,150)
shape = (300,300)

un = union(a,b,shape)
inter = intersection(a,b,shape)

# aaa = 1