import scipy.stats as stats
import pandas as pd
import numpy as np
import os
import time
import datetime

import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import statistics

def calculate_spearman_correlation(X, Y):
    return stats.spearmanr(X, Y)[0]
def calculate_spearman_correlation_p(X, Y):
    return stats.spearmanr(X, Y)[1]

def closest(mylist, Number):
    answer = []
    for i in mylist:
        answer.append(abs(Number-i))
    return answer.index(min(answer))


def flatten_list(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, list):
            result.extend(flatten_list(element))
        else:
            result.append(element)
    return result

keys = ['stride','step length', 'step speed', 'gait cycle', 'time variability','regularity lr', 'walking speed','arm rate', 'max angle']
a_01 = dict(zip(keys, [[] for _ in range(len(keys))]))
b_01 = dict(zip(keys, [[] for _ in range(len(keys))]))
a_02 = dict(zip(keys, [[] for _ in range(len(keys))]))
b_02 = dict(zip(keys, [[] for _ in range(len(keys))]))

all_l = dict(zip(keys, [[] for _ in range(len(keys))]))
all_m = dict(zip(keys, [[] for _ in range(len(keys))]))
all_h = dict(zip(keys, [[] for _ in range(len(keys))]))

path = r'F:\openpose_output'

no_list = ['a_092515']
pvalue = []
correlation = []

final_rate = []
final_x = []

shapiro_p = []

for home, dirs, files in os.walk(path):
    dirs.sort()
    for dir_name in dirs:
        #print(dir_name)

        if dir_name[:8] in no_list:
            continue

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
                all_l['stride'].append(stride.loc[:,'stride'].tolist())
                a_01['step length'].append(step_length.loc[:,'step length'].mean())
                all_l['step length'].append(step_length.loc[:,'step length'].tolist())
                a_01['step speed'].append(step_speed.loc[:,'step speed'].mean())
                all_l['step speed'].append(step_speed.loc[:,'step speed'].tolist())
                a_01['gait cycle'].append(gait_cycle.loc[:,'gait cycle'].mean())
                all_l['gait cycle'].append(gait_cycle.loc[:,'gait cycle'].tolist())
                a_01['time variability'].append(gait_cycle.loc[:,'gait cycle'].std())
                all_l['time variability'].append(gait_cycle.loc[:,'gait cycle'].std())
                a_01['regularity lr'].append(regu_lr.loc[:, 'regularity'].mean())
                all_l['regularity lr'].append(regu_lr.loc[:, 'regularity'].tolist())
                a_01['walking speed'].append(walking_speed.loc[:,'walking speed'].mean())
                all_l['walking speed'].append(walking_speed.loc[:,'walking speed'].tolist())
                #a_01['arm rate'].append(arm_rate.loc[:, 'arm'].mean())
                a_01['arm rate'].append(armangle.mean())
                all_l['arm rate'].append(armangle.tolist())
                a_01['max angle'].append(max_angle['max angle'].mean())
                all_l['max angle'].append(max_angle['max angle'].tolist())
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

                all_m['stride'].append(stride.loc[:, 'stride'].tolist())
                all_m['step length'].append(step_length.loc[:, 'step length'].tolist())
                all_m['step speed'].append(step_speed.loc[:, 'step speed'].tolist())
                all_m['gait cycle'].append(gait_cycle.loc[:, 'gait cycle'].tolist())
                all_m['time variability'].append(gait_cycle.loc[:, 'gait cycle'].std())
                all_m['regularity lr'].append(regu_lr.loc[:, 'regularity'].tolist())
                all_m['walking speed'].append(walking_speed.loc[:, 'walking speed'].tolist())
                all_m['arm rate'].append(armangle.tolist())
                all_m['max angle'].append(max_angle['max angle'].tolist())
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

                all_l['stride'].append(stride.loc[:, 'stride'].tolist())
                all_l['step length'].append(step_length.loc[:, 'step length'].tolist())
                all_l['step speed'].append(step_speed.loc[:, 'step speed'].tolist())
                all_l['gait cycle'].append(gait_cycle.loc[:, 'gait cycle'].tolist())
                all_l['time variability'].append(gait_cycle.loc[:, 'gait cycle'].std())
                all_l['regularity lr'].append(regu_lr.loc[:, 'regularity'].tolist())
                all_l['walking speed'].append(walking_speed.loc[:, 'walking speed'].tolist())
                all_l['arm rate'].append(armangle.tolist())
                all_l['max angle'].append(max_angle['max angle'].tolist())
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

                all_h['stride'].append(stride.loc[:, 'stride'].tolist())
                all_h['step length'].append(step_length.loc[:, 'step length'].tolist())
                all_h['step speed'].append(step_speed.loc[:, 'step speed'].tolist())
                all_h['gait cycle'].append(gait_cycle.loc[:, 'gait cycle'].tolist())
                all_h['time variability'].append(gait_cycle.loc[:, 'gait cycle'].std())
                all_h['regularity lr'].append(regu_lr.loc[:, 'regularity'].tolist())
                all_h['walking speed'].append(walking_speed.loc[:, 'walking speed'].tolist())
                all_h['arm rate'].append(armangle.tolist())
                all_h['max angle'].append(max_angle['max angle'].tolist())

        #if dir_name[0] == 'b' and dir_name[-1] == '2':

#a_01 = sum(a_01,[])
#a_02 = sum(a_02,[])
#b_01 = sum(b_01,[])
#b_02 = sum(b_02,[])

for i in range(len(keys)):
    all_l[keys[i]] = flatten_list(all_l[keys[i]])
    all_m[keys[i]] = flatten_list(all_m[keys[i]])
    all_h[keys[i]] = flatten_list(all_h[keys[i]])
    print(keys[i])
    #print(a_01[keys[i]])
    #print(a_02[keys[i]])
    print('mean')
    print(sum(all_l[keys[i]]) / len(all_l[keys[i]]))
    print(sum(all_m[keys[i]]) / len(all_m[keys[i]]))
    print(sum(all_h[keys[i]]) / len(all_h[keys[i]]))
    print('sd')
    print(statistics.stdev(all_l[keys[i]]))
    print(statistics.stdev(all_m[keys[i]]))
    print(statistics.stdev(all_h[keys[i]]))
    print('u-test')
    print(mannwhitneyu(all_m[keys[i]],all_h[keys[i]]))
    print(mannwhitneyu(a_01[keys[i]] + b_01[keys[i]],b_02[keys[i]]))
    print(mannwhitneyu(a_02[keys[i]],b_02[keys[i]]))
    print(mannwhitneyu(a_01[keys[i]],b_01[keys[i]]))

    df_all = pd.DataFrame({'values': a_01[keys[i]] + b_01[keys[i]] + a_02[keys[i]] + b_02[keys[i]],
                           'cateory': [1]*len(a_01[keys[i]] + b_01[keys[i]]) + [2]*len(a_02[keys[i]]) + [3]*len(b_02[keys[i]])})

    df_all.to_csv(os.path.join(path, keys[i] + '_all.csv'), index=False)



