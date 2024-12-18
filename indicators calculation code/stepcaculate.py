import os.path
import pickle
import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


def cacustep(dir_path, load_path):
    walk_ = []
    time_ = []

    #dir_path = 'F:/openpose_output/b_100112_yt_02'
    #load_path = os.path.join(dir_path, '100112_yt_02.pkl')
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    l = len(data)
    for i in range(l):
        joints_now = data[i]['skeleton'][1]
        #if (data[i]['skeleton'][9][2] < 0.7 or data[i]['skeleton'][10][2] < 0.7 or data[i]['skeleton'][11][2] < 0.7)\
            #and (data[i]['skeleton'][12][2] < 0.7 or data[i]['skeleton'][13][2] < 0.7 or data[i]['skeleton'][14][2] < 0.7 ):
        if data[i]['skeleton'][11][2] < 0.7 or data[i]['skeleton'][14][2] < 0.7:
            continue
        if joints_now[0] >= 160 and joints_now[0] <=800:
            walk_.append(data[i])
            time_.append(i)


    #walk = walk_

    walk_11_value = []
    walk_14_value = []
    for i in range(len(walk_)):
        walk_11_value.append(walk_[i]['skeleton'][11][0])
        walk_14_value.append(walk_[i]['skeleton'][14][0])

    joind_11_mean =np.mean(walk_11_value)
    joind_11_std =np.std(walk_11_value)

    joind_14_mean = np.mean(np.array(walk_14_value))
    joind_14_std = np.std(np.array(walk_14_value))

    walk = []
    time = []

    for i in range(len(walk_)):
        if i!=0 and ((walk_11_value[i] > joind_11_mean - 3 * joind_11_std and walk_14_value[i] > joind_14_mean - 3 * joind_14_std) or \
                (walk_11_value[i] < joind_11_mean + 3 * joind_11_std and walk_14_value[i] < joind_14_mean + 3 * joind_14_std)) and \
                (walk_11_value[i] != walk_11_value[i-1] and walk_14_value[i] != walk_14_value[i-1]):
            walk.append(walk_[i])
            time.append(time_[i])


    walk_ = walk
    time_ = time
    walk = []
    time = []
    for i in range(len(walk_)):
        if i >= 10 and i < len(walk_) - 10:
            if (walk_[i]['skeleton'][1][0] - walk_[i-10]['skeleton'][1][0]) * (walk_[i+10]['skeleton'][1][0] - walk_[i]['skeleton'][1][0]) >= 0:
                walk.append(walk_[i])
                time.append(time_[i])
            #else:

                #print(str(i) + ' ' + '-1')
        else:
            walk.append(walk_[i])
            time.append(time_[i])


    # print(len(walk))
    # for i in range(50) :
    #     cv2.imshow('walk frame', walk[random.randint(0,len(walk))]['frame'])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    # for i in range(20) :
    #     cv2.imshow('down frame', down[random.randint(0,len(down))]['frame'])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    steptride = []
    steptime = []
    for i in range(len(walk)):
        #print(data[i]['id'])
        #if i % 5 == 0 and abs((walk[i]['skeleton'][11][0]) - walk[i-1]['skeleton'][14][0]) < 200:
        if i % 5 == 0 and i != 0:
            steptride.append(abs((walk[i]['skeleton'][11][0]) - walk[i-1]['skeleton'][14][0]))
            steptime.append(time[i])
    #
    # walk = remove_elements(walk, steptride)
    #
    # st = []
    # for i in range(len(walk)):
    #     if i % 5 == 0:
    #         st.append(abs((walk[i]['skeleton'][11][0]) - walk[i-1]['skeleton'][14][0]))
    '''
    stride = []
    for i in range(len(steptride)):
        if i > 10 and i < len(steptride) - 10:
            if steptride[i] > max(steptride[i-10 : i-1]) and steptride[i] > max(steptride[i+1 : i+10]):
                stride.append(steptride[i])
    '''
    return steptride,steptime

if __name__ == "__main__":
    path = r'F:\openpose_output'
    for home, dirs, files in os.walk(path):
        dirs.sort()
        for dir_name in dirs:
            dir_path = os.path.join(path, dir_name)
            load_path = os.path.join(dir_path, dir_name[2:]+'.pkl')
            y,steptime = cacustep(dir_path, load_path)
            steptime = pd.DataFrame(steptime, columns=['gait cycle'])
            org_y = pd.DataFrame(y, columns=['step org'])
            final = pd.concat([steptime, org_y], axis=1)
            final.to_csv(os.path.join(dir_path, 'step_org.csv'))
            peak_1, _ = find_peaks(y, distance=5)
            peak_100 = [x for x in peak_1 if x <= 100]
            x = list(range(len(y)))
            plt.plot(x[:100], y[:100], c="r", alpha=0.5)
            plt.plot(peak_100, [y[i] for i in peak_100], "o")
            plt.title(dir_name)
            plt.show()
            y = pd.DataFrame([y[i] for i in peak_1], columns=['step length'])
            y.to_csv(os.path.join(dir_path, 'step_length.csv'))
            print(dir_name + ' finish')


