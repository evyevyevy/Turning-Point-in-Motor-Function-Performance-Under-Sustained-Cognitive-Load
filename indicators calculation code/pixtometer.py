import numpy as np
import os
import pandas as pd

path = r'F:\openpose_output'
for home, dirs, files in os.walk(path):
    dirs.sort()
    for dir_name in dirs:
        stepdir = os.path.join(path, dir_name)

        stride = pd.read_csv(os.path.join(stepdir, 'pixel_stride.csv'))

        step_length = pd.read_csv(os.path.join(stepdir, 'pixel_step_length.csv'))
        step_speed = pd.read_csv(os.path.join(stepdir, 'pixel_step_speed.csv'))
        gait_cycle = pd.read_csv(os.path.join(stepdir, 'pixel_gait_cycle.csv'))

        walking_speed = pd.read_csv(os.path.join(stepdir, 'pixel_walking_speed.csv'))

        stride['org stride'] = stride['stride']
        stride['stride'] = stride['org stride'] * (5.314 / 746.075)
        stride.to_csv(os.path.join(stepdir, 'new_stride.csv'))

        step_length['org step length'] = step_length['step length']
        step_length['step length'] = step_length['org step length'] * (5.314 / 746.075)
        step_length.to_csv(os.path.join(stepdir,'new_step_length.csv'))

        step_speed['org step speed'] = step_speed['step speed']
        step_speed['step speed'] = step_speed['org step speed'] * (5.314 / 746.075) * 75
        step_speed.to_csv(os.path.join(stepdir, 'new_step_speed.csv'))

        gait_cycle['org gait cycle'] = gait_cycle['gait cycle']
        gait_cycle['gait cycle'] = gait_cycle['org gait cycle'] / 75
        gait_cycle.to_csv(os.path.join(stepdir, 'new_gait_cycle.csv'))

        walking_speed['org walking speed'] = walking_speed['walking speed']
        walking_speed['walking speed'] = walking_speed['org walking speed'] * (5.314 / 746.075) * 75
        walking_speed.to_csv(os.path.join(stepdir, 'walking_speed.csv'))
