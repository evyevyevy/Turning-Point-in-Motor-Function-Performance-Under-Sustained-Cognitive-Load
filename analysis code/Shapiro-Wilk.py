

import scipy.stats as stats
import pandas as pd
import numpy as np
import os
import time
import datetime

import scipy.stats as stats
from scipy.stats import ttest_ind


def calculate_spearman_correlation(X, Y):
    return stats.spearmanr(X, Y)[0]
def calculate_spearman_correlation_p(X, Y):
    return stats.spearmanr(X, Y)[1]

def closest(mylist, Number):
    answer = []
    for i in mylist:
        answer.append(abs(Number-i))
    return answer.index(min(answer))



keys = ['stride','step length', 'step speed', 'gait cycle', 'time variability'\
        'regularity lr', 'walking speed','arm rate', 'max angle']
a_01 = dict(zip(keys, [[] for _ in range(len(keys))]))
b_01 = dict(zip(keys, [[] for _ in range(len(keys))]))
a_02 = dict(zip(keys, [[] for _ in range(len(keys))]))
b_02 = dict(zip(keys, [[] for _ in range(len(keys))]))

path = r'G:\openpose_output'


pvalue = []
correlation = []

final_rate = []
final_x = []

shapiro_p = []

for home, dirs, files in os.walk(path):
    dirs.sort()
    for dir_name in dirs:
        #print(dir_name)

        #if dir_name[2:-3] in no_list:
            #continue

        stepdir = os.path.join(path, dir_name)
        stride = pd.read_csv(os.path.join(stepdir, 'new_stride.csv'))
        step_length = pd.read_csv(os.path.join(stepdir, 'new_step_length.csv'))
        step_speed = pd.read_csv(os.path.join(stepdir, 'new_step_speed.csv'))
        gait_cycle = pd.read_csv(os.path.join(stepdir, 'new_gait_cycle.csv'))
        regu_lr = pd.read_csv(os.path.join(stepdir, 'new_regularity.csv'))
        walking_speed = pd.read_csv(os.path.join(stepdir, 'new_walking_speed.csv'))
        arm_rate = pd.read_csv(os.path.join(stepdir, 'new_arm_rate.csv'))
        max_angle = pd.read_csv(os.path.join(stepdir, 'new_max_angle.csv'))



        pupildir = os.path.join(r'H:\cleaned_data\Pupil', dir_name)
        pupil = pd.read_csv(os.path.join(pupildir, 'diameter_.csv'))
        pupil_time_ = pupil.loc[:,'timestamp'].values.flatten().tolist()
        pupil_dia_ = pupil.loc[:,'rate'].values.flatten().tolist()

        pupil_time = []
        pupil_dia = []

        #armmean = arm_rate.loc[:, 'arm'].mean()
        #armstd = arm_rate.loc[:, 'arm'].std()
        armangle = []
        for x in np.array(arm_rate.loc[:,'arm']) :
            #if x > armmean - 3*armstd and x < armmean + 3*armstd:
            if x > 1:
                armangle.append(1/x)
            else:
                armangle.append(x)
        armangle = np.array(armangle)
        armmean = armangle.mean()
        armstd = armangle.std()
        angle1 = [x for x in armangle if x > armmean - 3*armstd and x < armmean + 3 * armstd]
        armangle = np.array(angle1)


        for i in range(len(pupil_dia_)):
            if pupil_dia_[i] < 1:
                pupil_dia.append(pupil_dia_[i])
                pupil_time.append(pupil_time_[i])

        #print(step_length.loc[:,'step length'].mean())

        if dir_name[0] == 'a':
            if dir_name[-1] == '1':
                a_01['stride'].append(stride.loc[:,'stride'].mean())
                a_01['step length'].append(step_length.loc[:,'step length'].mean())
                a_01['step speed'].append(step_speed.loc[:,'step speed'].mean())
                a_01['gait cycle'].append(gait_cycle.loc[:,'gait cycle'].mean())
                a_01['time variability'].append(gait_cycle.loc[:,'gait cycle'].std())
                a_01['regularity lr'].append(regu_lr.loc[:, 'regularity'].mean())
                a_01['walking speed'].append(walking_speed.loc[:,'walking speed'].mean())
                #a_01['arm rate'].append(arm_rate.loc[:, 'arm'].mean())
                a_01['arm rate'].append(armangle.mean())
                a_01['max angle'].append(max_angle['max angle'].mean())
            else:
                a_02['stride'].append(stride.loc[:, 'stride'].mean())
                a_02['step length'].append(step_length.loc[:,'step length'].mean())
                a_02['step speed'].append(step_speed.loc[:,'step speed'].mean())
                a_02['gait cycle'].append(gait_cycle.loc[:,'gait cycle'].mean())
                a_02['time variability'].append(gait_cycle.loc[:,'gait cycle'].std())
                a_02['regularity lr'].append(regu_lr.loc[:, 'regularity'].mean())
                a_02['walking speed'].append(walking_speed.loc[:,'walking speed'].mean())
                #a_02['arm rate'].append(arm_rate.loc[:, 'arm'].mean())
                a_02['arm rate'].append(armangle.mean())
                a_02['max angle'].append(max_angle['max angle'].mean())
        else:
            if dir_name[-1] == '1':
                b_01['stride'].append(stride.loc[:, 'stride'].mean())
                b_01['step length'].append(step_length.loc[:,'step length'].mean())
                b_01['step speed'].append(step_speed.loc[:,'step speed'].mean())
                b_01['gait cycle'].append(gait_cycle.loc[:,'gait cycle'].mean())
                b_01['time variability'].append(gait_cycle.loc[:,'gait cycle'].std())
                b_01['regularity lr'].append(regu_lr.loc[:, 'regularity'].mean())
                b_01['walking speed'].append(walking_speed.loc[:,'walking speed'].mean())
                #b_01['arm rate'].append(arm_rate.loc[:, 'arm'].mean())
                b_01['arm rate'].append(armangle.mean())
                b_01['max angle'].append(max_angle['max angle'].mean())
            else:
                b_02['stride'].append(stride.loc[:, 'stride'].mean())
                b_02['step length'].append(step_length.loc[:,'step length'].mean())
                b_02['step speed'].append(step_speed.loc[:,'step speed'].mean())
                b_02['gait cycle'].append(gait_cycle.loc[:,'gait cycle'].mean())
                b_02['time variability'].append(gait_cycle.loc[:,'gait cycle'].std())
                b_02['regularity lr'].append(regu_lr.loc[:, 'regularity'].mean())
                b_02['walking speed'].append(walking_speed.loc[:,'walking speed'].mean())
                #b_02['arm rate'].append(arm_rate.loc[:, 'arm'].mean())
                b_02['arm rate'].append(armangle.mean())
                b_02['max angle'].append(max_angle['max angle'].mean())


p_value = []

for i in range(len(keys)):
    print(keys[i])
    #print(a_01[keys[i]])
    #print(a_02[keys[i]])
    # p_value.append(stats.shapiro(a_01[keys[i]])[1])
    # p_value.append(stats.shapiro(a_02[keys[i]])[1])
    # p_value.append(stats.shapiro(b_01[keys[i]])[1])
    # p_value.append(stats.shapiro(b_02[keys[i]])[1])

    print(stats.shapiro(a_01[keys[i]])[1])
    print(stats.shapiro(a_02[keys[i]])[1])
    print(stats.shapiro(b_01[keys[i]])[1])
    print(stats.shapiro(b_02[keys[i]])[1])


