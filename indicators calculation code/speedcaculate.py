import os.path
import pickle
import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt
import pandas as pd


def cacuspeed(dir_path, load_path):
    walk_ = []
    time_ = []

    # dir_path = 'F:/openpose_output/b_100112_yt_02'
    # load_path = os.path.join(dir_path, '100112_yt_02.pkl')
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    l = len(data)
    for i in range(l):
        joints_now = data[i]['skeleton'][1]
        # if (data[i]['skeleton'][9][2] < 0.7 or data[i]['skeleton'][10][2] < 0.7 or data[i]['skeleton'][11][2] < 0.7)\
        # and (data[i]['skeleton'][12][2] < 0.7 or data[i]['skeleton'][13][2] < 0.7 or data[i]['skeleton'][14][2] < 0.7 ):
        if data[i]['skeleton'][8][2] < 0.7 or data[i]['skeleton'][1][2] < 0.7:
            continue
        if joints_now[0] >= 160 and joints_now[0] <= 800:
            walk_.append(data[i])
            time_.append(i)

    walk_8_value = []
    for i in range(len(walk_)):
        walk_8_value.append(walk_[i]['skeleton'][8][0])

    joind_8_mean = np.mean(walk_8_value)
    joind_8_std = np.std(walk_8_value)

    walk = []
    time = []

    for i in range(len(walk_)):
        if i != 0 and ((walk_8_value[i] > joind_8_mean - 3 * joind_8_std)  or \
                       (walk_8_value[i] < joind_8_mean + 3 * joind_8_std) and \
                walk_8_value[i] != walk_8_value[i - 1]):
            walk.append(walk_[i])
            time.append(time_[i])


    walk_ = walk
    time_ = time
    walk = []
    time = []

    for i in range(len(walk_)):
        if i >= 30 and i < len(walk_) - 30:
            if (walk_[i]['skeleton'][1][0] - walk_[i-30]['skeleton'][1][0]) * (walk_[i+30]['skeleton'][1][0] - walk_[i]['skeleton'][1][0]) > 0:
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

    speed = []
    speed_time = []

    for i in range(len(walk)) :
        if i > 30 and i % 20 == 0:
            delta = abs(walk[i]['skeleton'][8][0] - walk[i-30]['skeleton'][8][0])
            if delta == 0:
                continue
            speed.append(delta / (time[i] - time[i-30]))
            speed_time.append(time[i])

    return speed,speed_time

if __name__ == "__main__":
    path = r'F:\openpose_output'
    for home, dirs, files in os.walk(path):
        dirs.sort()
        for dir_name in dirs:
            dir_path = os.path.join(path, dir_name)
            load_path = os.path.join(dir_path, dir_name[2:] + '.pkl')
            y,time = cacuspeed(dir_path,load_path)
            x = range(len(y))
            #plt.scatter(x,y, c="r", alpha=0.5)
            #plt.show()
            time = pd.DataFrame(time, columns=['time'])
            y = pd.DataFrame(y, columns=['walking speed'])
            final = pd.concat([time, y], axis=1)
            final.to_csv(os.path.join(dir_path, 'walking_speed.csv'))
