import numpy as np
import pandas as pd
import os
import datetime
import pickle

def OutlierDetection(df):

    u = df['diameter'].mean()

    std = df['diameter'].std()

    data_c = df[np.abs(df['diameter'] - u) <= 3 * std]

    return data_c

def convert_time(x, start_time):
    delta_ = (datetime.datetime.fromtimestamp(x) - datetime.datetime.fromtimestamp(
                    start_time)).total_seconds() * 1000
    return delta_


def remove_outliers(df, column_name):
    mean_val = df[column_name].mean()
    std_val = df[column_name].std()

    upper_bound = mean_val + 3 * std_val
    lower_bound = mean_val - 3 * std_val

    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]



if __name__ == "__main__":
    path = r'H:\cleaned_data\Pupil'
    accuracy = []


    for home, dirs, files in os.walk(path):
        dirs.sort()
        for dir_name in dirs:
            pupildir = os.path.join(path, dir_name)

            pupil_data = pd.read_csv(os.path.join(pupildir, 'pupil_positions.csv'))
            confidence = np.array(pupil_data['confidence'])

            diameter_ = []
            for i in range(len(confidence)):
                if i!=0 and confidence[i] >= 0.8 and pupil_data.loc[i, 'diameter']:
                    diameter_.append((pupil_data.loc[i, 'pupil_timestamp'], pupil_data.loc[i, 'diameter']))
                    accuracy.append(confidence[i])

            if len(diameter_) < 6000:
                diameter_ = []
                for i in range(len(confidence)):
                    if confidence[i] >= 0.35 and pupil_data.loc[i, 'diameter'] != 0:
                        diameter_.append((pupil_data.loc[i, 'pupil_timestamp'], pupil_data.loc[i, 'diameter']))
                        accuracy.append(confidence[i])
    print(sum(accuracy) / len(accuracy))

