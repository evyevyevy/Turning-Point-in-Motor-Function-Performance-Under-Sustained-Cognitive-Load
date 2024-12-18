import os.path

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
key_path = r'F:\openpose_output'
pupil_path = r'H:\cleaned_data\Pupil'

def convert_time(x, start_time):
    delta_ = (datetime.datetime.fromtimestamp(x) - datetime.datetime.fromtimestamp(
                    start_time)).total_seconds() * 1000
    return delta_

def convert_video(x):
    return x / 72 * 1000

def remove_outliers_3(df, column_name):

    mean_val = df[column_name].mean()
    std_val = df[column_name].std()

    upper_bound = mean_val + 3 * std_val
    lower_bound = mean_val - 3 * std_val

    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

correlation = []

for home, dirs, files in os.walk(key_path):
    dirs.sort()
    for dir_name in dirs:
        new_key_path = os.path.join(key_path, dir_name)
        data = pd.read_csv(os.path.join(new_key_path, 'new_stride.csv'))
        stride = data['stride'].values.flatten().tolist()
        time = data['time'].values.flatten().tolist()
        regu_lr = []
        lr_time = []
        for i in range(len(stride)):
            if i != 0:
                _ = stride[i] / stride[i-1]
                if _ >= 1:
                    _ = 1 / _
                regu_lr.append(_)
                lr_time.append(time[i])
        regu_lr = pd.DataFrame(regu_lr, columns=['regularity'])
        lr_time = pd.DataFrame(lr_time, columns=['time'])
        final = pd.concat([lr_time, regu_lr], axis=1)
        #final = remove_outliers_3(final, 'regurality')
        final.to_csv(os.path.join(new_key_path, 'new_regularity.csv'))