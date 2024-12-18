import scipy.stats as stats
import pandas as pd
import numpy as np
import os
from scipy.stats import kruskal

a_01 = []
b_01 = []
a_02 = []
b_02 = []

all_l = []
all_m = []
all_h = []



def flatten_list(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, list):
            result.extend(flatten_list(element))
        else:
            result.append(element)
    return result


no_list = ['a_092306','a_080201','a_081002', 'b_092410','b_100106','a_092506','a_080212']

path = r'H:\cleaned_data\Pupil'
for home, dirs, files in os.walk(path):
    dirs.sort()
    for dir_name in dirs:
        #print(dir_name)
        if dir_name[:8] in no_list:
            continue
        pupildir = os.path.join(path, dir_name)
        if dir_name[0] == 'a' :
            pupil_dia = pd.read_csv(os.path.join(pupildir, 'new_diameter_2.csv'))
        else:
            pupil_dia = pd.read_csv(os.path.join(pupildir, 'new_diameter_.csv'))
        mean = pupil_dia['rate'].mean()

        #pupil_dia = pupil_dia[(pupil_dia['rate'] < mean + 3 * std) & ((pupil_dia['rate'] > mean - 3 * std))]
        rate = pupil_dia['rate'].tolist()
        # pupil = pupil_dia[(pupil_dia['rate'] < 1) & ((pupil_dia['rate'] > -1))]
        # pupil["rate_rolling"] = pupil['rate'].rolling(window=180).mean()
        # #rate = [x for x in rate if x < 1]
        #
        # mean_rate = pupil["rate_rolling"].mean()
        #mean_rate = pupil_dia['rate'].mean()
        if dir_name[0] == 'a':
            if dir_name[-1] == '1':
                a_01.append(mean)
                #all_l.append(pupil_dia['rate'].tolist())

            else:
                a_02.append(mean)
                #all_m.append(pupil_dia['rate'].tolist())

        else:
            if dir_name[-1] == '1':
                b_01.append(mean)
                #all_l.append(pupil_dia['rate'].tolist())

            else:
                b_02.append(mean)
                #all_h.append(pupil_dia['rate'].tolist())





# all_l = flatten_list(all_l)
# all_m = flatten_list(all_m)
# all_h = flatten_list(all_h)


print('low: {}'.format(sum(a_01 + b_01 )/ len(a_01 + b_01)))

print('medium: {}'.format(sum(a_02)/ len(a_02)))

print('high: {}'.format(sum(b_02)/len(b_02)))
stat, p_value = kruskal(a_01 + b_01, a_02, b_02)
print(f'P-Value: {p_value}')
print(stats.mannwhitneyu(a_01 + b_01,a_02,alternative='two-sided'))
print(stats.mannwhitneyu(a_01 + b_01,b_02,alternative='two-sided'))
print(stats.mannwhitneyu(a_02,b_02,alternative='two-sided'))

print()
print(stats.shapiro(a_01 + b_01))
print(stats.shapiro(a_02))
#print(stats.shapiro(b_01))
print(stats.shapiro(b_02))
#print(stats.mannwhitneyu(a_01,a_01,alternative='two-sided'))

