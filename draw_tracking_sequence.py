import cv2
import numpy as np
import os
import re
import sys


#RECTANGLE FUNCTIONS
#Inputs should be (x0,y0,x1,y1) = upper left and lower right
#Shape should be (lines,cols) (NOT (width,height)=(cols,lines) )
def union(a,b, imshape):
    #Doubles size of image (to be robust to supplied negative coords, as is the case in some GT txt rectangles)
    imshape = (imshape[0]*2,imshape[1]*2)
    offset_i = (imshape[0]//4)
    offset_j = (imshape[1]//4)

    #Prepare to draw two rectangles
    a = (a[0] + offset_j, a[1] + offset_i, a[2] + offset_j, a[3] + offset_i)
    b = (b[0] + offset_j, b[1] + offset_i, b[2] + offset_j, b[3] + offset_i)


    #Draw them together
    zeroed = np.zeros(imshape, dtype = np.bool)
    zeroed[ a[1]:a[3], a[0]:a[2] ] = True
    zeroed[ b[1]:b[3], b[0]:b[2] ] = True

    #Get sum
    return float(np.sum(zeroed))

def intersection(a,b, imshape):

    #Doubles size of image (to be robust to supplied negative coords, as is the case in some GT txt rectangles)
    imshape = (imshape[0]*2,imshape[1]*2)
    offset_i = (imshape[0]//4)
    offset_j = (imshape[1]//4)

    #Prepare to draw two rectangles
    a = (a[0] + offset_j, a[1] + offset_i, a[2] + offset_j, a[3] + offset_i)
    b = (b[0] + offset_j, b[1] + offset_i, b[2] + offset_j, b[3] + offset_i)

    #Draw them in separate images
    zeroed_one = np.zeros(imshape, dtype = np.bool)
    zeroed_one[ a[1]:a[3], a[0]:a[2] ] = True

    zeroed_two = np.zeros(imshape, dtype = np.bool)
    zeroed_two[ b[1]:b[3], b[0]:b[2] ] = True

    #Intersect them
    result = np.logical_and(zeroed_one, zeroed_two)

    #Get sum
    return float(np.sum(result))

#TRACKER INITIALIZATION
#Should be re-initialized for every flaw
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

#MAIN FUNCTION
def run(args):

    #GENERAL VARIABLES
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.') #version

    jpg_pattern = re.compile('\w*\.jpg') #regex pattern to filter jpg

    basedir = args[1]  #'PD8-files/car1' -> directory where to read images from
    txt_filename = args[2] #'PD8-files/gtcar1.txt' -> directory where to read BB from

    all_files = [ os.path.join(basedir,i) for i in os.listdir(basedir) if jpg_pattern.search(i) is not None] #All images to be read

    file_obj  = open(txt_filename, 'r') #Coordinates

    # tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW']
    # GOTURN does not work (bugged implementation)
    tracker_type = args[3]

    out_txt = args[4] #filename where to write results

    out_file_obj  = open(out_txt, 'w') #file obj where to write results

    draw = (args[5] == "True") #whether to draw or simply show pct

    # MAIN LOOP VARIABLES
    N = 0
    F = 0

    tracker_initialization_pending = True #flag to reinitialize tracker

    valid_frame = True #track to decide if the frame should be counted in N (NaN frames are ignored)

    jacc_list = [] #list with Jaccards
    fps_list = [] #list with FPS

    # MAIN LOOP

    for idx,filename in enumerate(all_files):

        #Print progress percentage
        if not draw:
            os.system('clear')
            print( "%s: %f%%" % (out_txt, (float(idx + 1)/ len(all_files) ) * 100) )

        img = cv2.imread(filename, cv2.IMREAD_COLOR)

        #Read coordinates
        line = file_obj.readline()
        split = line.split(",") #Separate the line elems and store in array

        if 'NaN' not in split: #If there is no NaN, prepare to draw GT BB
            valid_frame = True #Frame is valid
            N = N + 1

            #BB coordinates
            x0 = int(float(split[0]))
            y0 = int(float(split[1]))
            x1 = int(float(split[2]))
            y1 = int(float(split[3]))

            cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),2)

            #If previous interactions failed, reinitialize tracker
            if tracker_initialization_pending:
                tracker = initialize_tracker(minor_ver, tracker_type)

                # To avoid generating exceptions
                # (if you try to init a tracker with negative coords it raises exceptions)
                # (GT txt contains negative coords)
                height, width, _ = img.shape
                c_x0 = min(max(x0, 0), width - 1)
                c_x1 = min(max(x1, 0), width - 1)
                c_y0 = min(max(y0, 0), height - 1)
                c_y1 = min(max(y1, 0), height - 1)

                # Init tracker
                ok = tracker.init(img, (c_x0,c_y0,c_x1-c_x0,c_y1-c_y0) )
                tracker_initialization_pending = False #Suspend pending initialization
                skip_tracking = True #Skip tracking on this frame (start on next)
        else:
            valid_frame = False #If there is NaN, do not count for metrics


        #Try to track
        if idx > 0 and not tracker_initialization_pending and not skip_tracking:
            # Start timer (for FPS metric)
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(img)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            if ok: # Draw bounding box if tracking succeeds
                fps_list.append(fps)
                # Draw tracking BB
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)

                #If GT present, calculate metrics
                if valid_frame:
                    ret1 = (p1[0], p1[1], p2[0], p2[1])
                    ret2 = (x0,y0,x1,y1)
                    shape = img.shape
                    un = union( ret1, ret2, shape )
                    inter = intersection( ret1, ret2, shape )

                    jacc = inter/un

                    jacc_list.append(jacc)

                    #If boxes do not overlap, count as failure
                    if jacc == 0.0:
                        F = F + 1
                        cv2.putText(img, "Tracking failure detected", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,255),1)
                        tracker_initialization_pending = True


            else : # Tracker could not return BB
                if valid_frame: #If GT is present, count as failure and reinitialize tracker on next frame
                    F = F + 1
                    cv2.putText(img, "Tracking failure detected", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,255),1)
                    tracker_initialization_pending = True
                else: #If GT is not present, ignore (the tracker does not commit mistakes if there is no GT)
                    cv2.putText(img, "Tracking failure detected (but frame is invalid)", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,255),1)

        elif skip_tracking:
            skip_tracking = False #Suspend skip tracking on next frames

        #Draw if needed
        if draw:
            cv2.imshow(tracker_type,img)

            a = cv2.waitKey(20) & 0xFF
            if a == ord('q'):
                break

    file_obj.close()


    #Calculate metrics
    np_jacc_list = np.array(jacc_list, dtype = np.float32)
    np_fps_list = np.array(fps_list, dtype = np.float32)

    mean_jacc = np.mean(np_jacc_list)
    std_jacc = np.std(np_jacc_list)

    mean_fps = np.mean(np_fps_list)
    std_fps = np.std(np_fps_list)
    M = F/N
    S = 30
    R = np.exp(-1*S*M)

    #Write metrics on files
    out_file_obj.write("Mean Jacc: %f, Std Jacc: %f\n"%(mean_jacc,std_jacc))
    out_file_obj.write("Mean FPS: %f, Std FPS: %f\n"%(mean_fps,std_fps))
    out_file_obj.write("Robustness: %f, Flaws: %d, Sequence: %d\n"%(R,F,N))

    out_file_obj.close()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run(sys.argv)
