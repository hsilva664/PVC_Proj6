import os
import sys
import draw_tracking_sequence

basedir_list = ['PD8-files/car1','PD8-files/car2','PD8-files/Volkswagen']
txt_filename_list = ['PD8-files/gtcar1.txt','PD8-files/volkswagen.txt']


tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW']
# GOTURN does not work (bugged implementation)
tracker_types = ['KCF']
draw = "False"
#draw = "True"
for i in range(len(basedir_list)):
    basedir = basedir_list[i]
    txt_filename = txt_filename_list[i]

    for j in range(len(tracker_types)):
        tracker_type = tracker_types[j]
        out_filename = 'results/' + tracker_type + '_' + str(i) + '.txt'

        in_args = [None, basedir, txt_filename, tracker_type, out_filename, draw]

        draw_tracking_sequence.run(in_args)
