import cv2
import numpy as np
import os
import re
import sys


#RECTANGLE FUNCTIONS
#Inputs should be (x0,y0,x1,y1)
#Shape should be (lines,cols)
def union(a,b, imshape):
    imshape = (imshape[0]*2,imshape[1]*2)
    offset_i = (imshape[0]//4)
    offset_j = (imshape[1]//4)

    a = (a[0] + offset_j, a[1] + offset_i, a[2] + offset_j, a[3] + offset_i)
    b = (b[0] + offset_j, b[1] + offset_i, b[2] + offset_j, b[3] + offset_i)

    zeroed = np.zeros(imshape, dtype = np.bool)    
    zeroed[ a[1]:a[3], a[0]:a[2] ] = True
    zeroed[ b[1]:b[3], b[0]:b[2] ] = True


    return float(np.sum(zeroed))

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

    return float(np.sum(result))

#TRACKER INITIALIZATION
def initialize_tracker(minor_ver, tracker_type):
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
    return tracker


def run(args):

    #GENERAL VARIABLES
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    jpg_pattern = re.compile('\w*\.jpg')
    
    basedir = args[1]  #'PD8-files/car1'
    txt_filename = args[2] #'PD8-files/gtcar1.txt'

    all_files = [ os.path.join(basedir,i) for i in os.listdir(basedir) if jpg_pattern.search(i) is not None]

    file_obj  = open(txt_filename, 'r')

    # tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW']
    tracker_type = args[3]

    out_txt = args[4]

    out_file_obj  = open(out_txt, 'w')

    # MAIN LOOP VARIABLES
    N = 0
    F = 0

    tracker_initialization_pending = True

    valid_frame = True

    jacc_list = []
    fps_list = []

    # MAIN LOOP

    for idx,filename in enumerate(all_files):

        img = cv2.imread(filename, cv2.IMREAD_COLOR)

        line = file_obj.readline()
        split = line.split(",")

        if 'NaN' not in split:
            valid_frame = True
            N = N + 1

            x0 = int(float(split[0]))
            y0 = int(float(split[1]))
            x1 = int(float(split[2]))
            y1 = int(float(split[3]))

            cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),2)

            if tracker_initialization_pending:
                tracker = initialize_tracker(minor_ver, tracker_type)
                ok = tracker.init(img, (x0,y0,x1-x0,y1-y0) )            
                tracker_initialization_pending = False
                skip_tracking = True
        else:
            valid_frame = False


        if idx > 0 and not tracker_initialization_pending and not skip_tracking:
            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(img)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            if ok:
                fps_list.append(fps)
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
                if valid_frame:
                    ret1 = (p1[0], p1[1], p2[0], p2[1])
                    ret2 = (x0,y0,x1,y1)
                    shape = img.shape
                    un = union( ret1, ret2, shape )
                    inter = intersection( ret1, ret2, shape )

                    aa = 1

                    jacc = inter/un

                    jacc_list.append(jacc)

                    if jacc == 0.0:
                        F = F + 1
                        cv2.putText(img, "Tracking failure detected", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,255),1)        
                        tracker_initialization_pending = True                                       
                    

            else :
                # Tracking failure
                if valid_frame:
                    F = F + 1
                    cv2.putText(img, "Tracking failure detected", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,255),1)        
                    tracker_initialization_pending = True
                else:
                    cv2.putText(img, "Tracking failure detected (but frame is invalid)", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)        

        elif skip_tracking:
            skip_tracking = False
        
        cv2.imshow('teste',img)

        a = cv2.waitKey(20) & 0xFF
        if a == ord('q'):
            break

    file_obj.close()  

    np_jacc_list = np.array(jacc_list, dtype = np.float32)
    np_fps_list = np.array(fps_list, dtype = np.float32)

    mean_jacc = np.mean(np_jacc_list)
    std_jacc = np.std(np_jacc_list)

    mean_fps = np.mean(np_fps_list)
    std_fps = np.std(np_fps_list)    

    M = F/N
    S = 30
    R = np.exp(-1*S*M)


    out_file_obj.write("Mean Jacc: %f, Std Jacc: %f\n"%(mean_jacc,std_jacc))
    out_file_obj.write("Mean FPS: %f, Std FPS: %f\n"%(mean_fps,std_fps))
    out_file_obj.write("Robustness: %f, Flaws: %d, Sequence: %d\n"%(R,F,N))

    out_file_obj.close()

if __name__ == "__main__":
    run(sys.argv)    