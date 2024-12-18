import os.path
import pickle
import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


def calculate_angle(A, B, C):

    AB = [B[0]-A[0], B[1]-A[1]]
    BC = [C[0]-B[0], C[1]-B[1]]


    dot_product = AB[0]*BC[0] + AB[1]*BC[1]
    norm_AB = math.sqrt(AB[0]**2 + AB[1]**2)
    norm_BC = math.sqrt(BC[0]**2 + BC[1]**2)


    angle = math.acos(dot_product / (norm_AB * norm_BC))


    angle = math.degrees(angle)

    if angle > 180:
        return angle - 180
    else:
        return angle

def cacustep(dir_path, load_path):
    updown_ = []
    walk_ = []
    time_ = []


    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    l = len(data)
    for i in range(l):
        joints_now = data[i]['skeleton'][1]
        #if (data[i]['skeleton'][9][2] < 0.7 or data[i]['skeleton'][10][2] < 0.7 or data[i]['skeleton'][11][2] < 0.7)\
            #and (data[i]['skeleton'][12][2] < 0.7 or data[i]['skeleton'][13][2] < 0.7 or data[i]['skeleton'][14][2] < 0.7 ):
        if data[i]['skeleton'][1][2] >= 0.7 and data[i]['skeleton'][5][2] >= 0.7 and\
                data[i]['skeleton'][6][2] >= 0.7  and data[i]['skeleton'][7][2] >= 0.7 and\
                joints_now[0] < 160:
            updown_.append(data[i])
            time_.append(i)
        if data[i]['skeleton'][1][2] >= 0.7 and data[i]['skeleton'][2][2] >= 0.7 and\
                data[i]['skeleton'][3][2] >= 0.7  and data[i]['skeleton'][4][2] >= 0.7 and\
                joints_now[0] > 800:
            updown_.append(data[i])
            time_.append(i)


    #walk = walk_

    langle = []
    rangle = []
    ltime = []
    rtime = []

    for i in range(len(updown_)):
        if updown_[i]['skeleton'][1][0] < 160:
            A = updown_[i]['skeleton'][5][:2]
            B = updown_[i]['skeleton'][6][:2]
            C = updown_[i]['skeleton'][7][:2]
            langle.append(calculate_angle(A,B,C))
            ltime.append(time_[i])
        elif updown_[i]['skeleton'][1][0] > 800:
            A = updown_[i]['skeleton'][2][:2]
            B = updown_[i]['skeleton'][3][:2]
            C = updown_[i]['skeleton'][4][:2]
            rangle.append(calculate_angle(A,B,C))
            rtime.append(time_[i])

    joind_l_mean = np.mean(langle)
    joind_l_std = np.std(langle)

    joind_r_mean = np.mean(np.array(rangle))
    joind_r_std = np.std(np.array(rangle))

    langle_c = []
    rangle_c = []
    ltime_c = []
    rtime_c = []

    for i in range(len(langle)):
        if i != 0 and i % 3 == 0 and (langle[i] > joind_l_mean - 3 * joind_l_std or \
                       langle[i] < joind_l_mean + 3 * joind_l_std ):
            langle_c.append(langle[i])
            ltime_c.append(ltime[i])

    for i in range(len(rangle)):
        if i != 0 and i % 3 == 0 and (rangle[i] > joind_r_mean - 3 * joind_r_std or \
                       rangle[i] < joind_r_mean + 3 * joind_r_std):
            rangle_c.append(rangle[i])
            rtime_c.append(rtime[i])


    return langle_c,rangle_c,ltime_c,rtime_c

if __name__ == "__main__":
    path = r'F:\openpose_output'
    for home, dirs, files in os.walk(path):
        dirs.sort()
        for dir_name in dirs:
            dir_path = os.path.join(path, dir_name)
            load_path = os.path.join(dir_path, dir_name[2:]+'.pkl')
            langle,rangle,ltime,rtime = cacustep(dir_path, load_path)

            ltime = pd.DataFrame(ltime, columns=['left time'])
            langle = pd.DataFrame(langle, columns=['left angle'])
            rtime = pd.DataFrame(rtime, columns=['right time'])
            rangle = pd.DataFrame(rangle, columns=['right angle'])
            final = pd.concat([ltime,langle,rtime,rangle], axis=1)
            final.to_csv(os.path.join(dir_path, 'arm_angle.csv'))
            #peak_1, _ = find_peaks(y, distance=5)
            #peak_100 = [x for x in peak_1 if x <= 100]
            y = langle['left angle'].tolist()
            x = list(range(len(y)))
            plt.plot(x[:100], y[:100], c="r", alpha=0.5)
            #plt.plot(peak_100, [y[i] for i in peak_100], "o")
            plt.title(dir_name)
            plt.show()
            #y = pd.DataFrame([y[i] for i in peak_1], columns=['step length'])
            #y.to_csv(os.path.join(dir_path, 'step_length.csv'))
            print(dir_name + ' finish')





