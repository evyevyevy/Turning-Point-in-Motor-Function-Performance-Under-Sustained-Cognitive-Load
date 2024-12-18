import os.path
import pickle
import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


def remove_outliers(df, column_name):

    mean_val = df[column_name].mean()
    std_val = df[column_name].std()

    upper_bound = mean_val + 3 * std_val
    lower_bound = mean_val - 3 * std_val

    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

if __name__ == "__main__":
    path = r'F:\openpose_output'
    for home, dirs, files in os.walk(path):
        dirs.sort()
        for dir_name in dirs:
            if dir_name[0] == 'a':
                continue
            dir_path = os.path.join(path, dir_name)
            if os.path.exists(os.path.join(dir_path, 'new_step_speed.csv')):
                continue
            #load_path = os.path.join(dir_path, dir_name[2:]+'.pkl')
            data = pd.read_csv(os.path.join(dir_path, 'step_org.csv'), index_col=0)
            data = data.set_index("step time")
            #data = data.head(200)


            distances = data['step org'].values
            peaks_refined, _ = find_peaks(distances, distance=1.5, prominence=30)

            if len(peaks_refined) < 160:
                peaks_refined, _ = find_peaks(distances)

            step_sizes_refined = distances[peaks_refined]

            stride = []
            time = []

            for i in range(len(peaks_refined)):
                if i != 0 and data.index[peaks_refined[i]] - data.index[peaks_refined[i-1]] < 100:
                    stride.append(step_sizes_refined[i])
                    time.append(data.index[peaks_refined[i]])
                elif len(stride) > 1 and data.index[peaks_refined[i]] - data.index[peaks_refined[i-1]] >= 100:
                    del stride[-1]
                    del time[-1]
            print(dir_name + ' ' + str(len(stride)))
            df_stride = pd.DataFrame(stride, columns=['stride'])
            df_time = pd.DataFrame(time, columns=['time'])
            df = pd.concat([df_time,df_stride], axis=1)
            df = remove_outliers(df, 'stride')
            df.to_csv(os.path.join(dir_path, 'new_stride.csv'))

            step_length = []
            l_time = []
            for i in range(len(peaks_refined)):
                if i != 0 and data.index[peaks_refined[i]] - data.index[peaks_refined[i-1]] < 100:
                    step_length.append(step_sizes_refined[i] + step_sizes_refined[i-1])
                    l_time.append(data.index[peaks_refined[i]])
                elif len(step_length) > 1 and data.index[peaks_refined[i]] - data.index[peaks_refined[i-1]] >= 100:
                    del step_length[-1]
                    del l_time[-1]
            df_step_length = pd.DataFrame(step_length, columns=['step length'])
            df_l_time = pd.DataFrame(l_time, columns=['time'])
            df = pd.concat([df_l_time, df_step_length], axis=1)
            df = remove_outliers(df,'step length')
            df.to_csv(os.path.join(dir_path, 'new_step_length.csv'))

            step_cycle = []
            c_time = []
            for i in range(len(peaks_refined)):
                if i != 0 and data.index[peaks_refined[i]] - data.index[peaks_refined[i - 1]] < 100:
                    step_cycle.append(data.index[peaks_refined[i]] - data.index[peaks_refined[i-1]])
                    c_time.append(data.index[peaks_refined[i]])
                elif len(step_cycle) > 1 and data.index[peaks_refined[i]] - data.index[peaks_refined[i - 1]] >= 100:
                    del step_cycle[-1]
                    del c_time[-1]
            df_step_cycle = pd.DataFrame(step_cycle, columns=['gait cycle'])
            df_c_time = pd.DataFrame(c_time, columns=['time'])
            df = pd.concat([df_c_time, df_step_cycle], axis=1)
            df = remove_outliers(df, 'gait cycle')
            df.to_csv(os.path.join(dir_path, 'new_gait_cycle.csv'))

            step_speed = []
            s_time = []
            for i in range(len(peaks_refined)):
                if i != 0 and data.index[peaks_refined[i]] - data.index[peaks_refined[i - 1]] < 100:
                    step_speed.append( (step_sizes_refined[i] + step_sizes_refined[i-1]) / (data.index[peaks_refined[i]] - data.index[peaks_refined[i-1]]))
                    s_time.append(data.index[peaks_refined[i]])
                elif len(step_speed) > 1 and data.index[peaks_refined[i]] - data.index[peaks_refined[i - 1]] >= 100:
                    del step_speed[-1]
                    del s_time[-1]
            df_step_speed = pd.DataFrame(step_speed, columns=['step speed'])
            df_s_time = pd.DataFrame(s_time, columns=['time'])
            df = pd.concat([df_s_time, df_step_speed], axis=1)
            df = remove_outliers(df, 'step speed')
            df.to_csv(os.path.join(dir_path, 'new_step_speed.csv'))

