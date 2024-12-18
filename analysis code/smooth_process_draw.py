import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def convert_video(x):
    return x / 72 * 1000

data1 = pd.read_csv(os.path.join(r'H:\cleaned_data\Pupil\a_092501_01', 'new_diameter_.csv'))
data1["rate_rolling"] = data1['rate'].rolling(window=160).mean()
data2 = pd.read_csv(os.path.join(r'H:\cleaned_data\Pupil\a_092501_02', 'new_diameter_.csv'))
data2["rate_rolling"] = data1['rate'].rolling(window=160).mean()


plt.figure(figsize=(10, 6))
#plt.plot(selected_data['converttime'], selected_data['rate'], label='original pupil data', color = '#2E5DA5')
plt.plot(data1['time'], data1['rate_rolling'], color = '#2E5DA5',label = 'low cognitive load')
plt.plot(data2['time'], data2['rate_rolling'], color = '#DE752D',label = 'moderate cognitive load')
plt.xlabel('Time (millesecond)',fontsize = 20, font = 'Arial')
plt.ylim(-0.3,0.3)
plt.ylabel('Pupil dilation ratio',fontsize = 20, font = 'Arial')
plt.legend(fontsize = 14)
plt.grid(True)
plt.savefig(r'G:\figure_nomal\figure-meter\pupil_time-1.png', dpi=300, format='png', bbox_inches='tight')
plt.show()

data1 = pd.read_csv(os.path.join(r'H:\cleaned_data\Pupil\b_093005_01', 'new_diameter_.csv'))
data1["rate_rolling"] = data1['rate'].rolling(window=160).mean()
data2 = pd.read_csv(os.path.join(r'H:\cleaned_data\Pupil\b_093005_02', 'new_diameter_.csv'))
data2["rate_rolling"] = data1['rate'].rolling(window=160).mean()


plt.figure(figsize=(10, 6))
#plt.plot(selected_data['converttime'], selected_data['rate'], label='original pupil data', color = '#2E5DA5')
plt.plot(data1['time'], data1['rate_rolling'], color = '#2E5DA5',label = 'low cognitive load')
plt.plot(data2['time'], data2['rate_rolling'], color = '#DE752D',label = 'high cognitive load')
plt.xlabel('Time (millesecond)',fontsize = 20, font = 'Arial')
plt.ylim(-0.3,0.3)
plt.ylabel('Pupil dilation ratio',fontsize = 20, font = 'Arial')
plt.legend(fontsize = 14)
plt.grid(True)
plt.savefig(r'G:\figure_nomal\figure-meter\pupil_time.png', dpi=300, format='png', bbox_inches='tight')
plt.show()

data = pd.read_csv(os.path.join(r'H:\cleaned_data\Pupil\a_072801_01', 'new_diameter_.csv'))
data["rate_rolling"] = data['rate'].rolling(window=180).mean()

selected_data = data.loc[500:1000]


plt.figure(figsize=(10, 6))
plt.plot(selected_data.index, selected_data['rate'], label='original pupil data', color = '#2E5DA5')
plt.plot(selected_data.index, selected_data['rate_rolling'], label='smoothed pupil data', color = '#DE752D')
plt.xlabel('Index',fontsize = 20, font = 'Arial')
plt.ylabel('Pupil dilation ratio',fontsize = 20, font = 'Arial')
plt.legend(fontsize = 14)
plt.grid(True)
plt.savefig(r'G:\figure_nomal\figure-meter\pupil_smoothed-1.png', dpi=300, format='png', bbox_inches='tight')
plt.show()

data = pd.read_csv(os.path.join(r'G:\openpose_output\a_072801_01', 'new_stride.csv'))
data['stride_rolling'] = data['stride'].rolling(window=15).mean()

selected_data = data.loc[30:80]


plt.figure(figsize=(10, 6))
plt.plot(selected_data.index, selected_data['stride'], label='original stride length data', color = '#2E5DA5')
plt.plot(selected_data.index, selected_data['stride_rolling'], label='smoothed stride length data', color = '#DE752D')
plt.xlabel('Index',fontsize = 20, font = 'Arial')
plt.ylabel('Stride length (m)',fontsize = 20, font = 'Arial')
plt.legend(fontsize = 14, loc='upper right')
plt.grid(True)
plt.savefig(r'G:\figure_nomal\figure-meter\stride_smoothed-1.png', dpi=300, format='png', bbox_inches='tight')
plt.show()

stride = pd.read_csv(os.path.join(r'G:\openpose_output\b_072802_02', 'new_stride.csv'))
stride['stride_rolling'] = stride['stride'].rolling(window=15).mean()
pupil = pd.read_csv(os.path.join(r'H:\cleaned_data\Pupil\b_072802_02', 'new_diameter_.csv'))
pupil["rate_rolling"] = pupil['rate'].rolling(window=180).mean()

stride['converttime'] = stride['time'].apply(lambda x:convert_video(x))
merge = pd.merge_asof(stride,pupil,left_on='converttime', right_on="time",direction="nearest")

selected_data = merge.loc[30:200]


plt.figure(figsize=(10, 6))
plt.plot(selected_data['converttime'], selected_data['rate_rolling'], label='pupil dilation', color = '#2E5DA5')
plt.plot(selected_data['converttime'], selected_data['stride_rolling'], label='stride length', color = '#DE752D')
plt.xlabel('Time (millisecond)',fontsize = 20,font = 'Arial')
plt.ylabel('Pupil and stride',fontsize = 20, font = 'Arial')
plt.legend(fontsize = 14, loc='upper right', framealpha=0.5)
plt.grid(True)
plt.savefig(r'G:\figure_nomal\figure-meter\time_pupil_strde-1.png', dpi=300, format='png', bbox_inches='tight')
plt.show()