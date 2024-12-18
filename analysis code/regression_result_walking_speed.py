import os.path
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
key_path = r'G:\openpose_output'
pupil_path = r'H:\cleaned_data\Pupil'
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import heapq
from matplotlib.ticker import FuncFormatter

def convert_time(x, start_time):
    delta_ = (datetime.datetime.fromtimestamp(x) - datetime.datetime.fromtimestamp(
                    start_time)).total_seconds() * 1000
    return delta_

def convert_video(x):
    return x / 72 * 1000

def remove_outliers(df, column_name):

    mean_val = df[column_name].mean()
    std_val = df[column_name].std()

    #upper_bound = mean_val + 3 * std_val
    lower_bound = mean_val - 2 * std_val

    return df[df[column_name] >= lower_bound]

def remove_outliers_3(df, column_name):

    mean_val = df[column_name].mean()
    std_val = df[column_name].std()

    upper_bound = mean_val + 3 * std_val
    lower_bound = mean_val - 3 * std_val

    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]


def calculate_p_values(x, y):

    coefficients = np.polyfit(x, y, 1)
    model = np.poly1d(coefficients)

    y_pred = model(x)
    residuals = y - y_pred

    n = len(x)
    p = 2
    df = n - p

    residual_std_error = np.sqrt(np.sum(residuals ** 2) / df)

    X = np.vstack([x, np.ones(len(x))]).T

    XtX_inv = np.linalg.inv(X.T @ X)
    stderrs = np.sqrt(np.diag(XtX_inv)) * residual_std_error

    t_stats = coefficients / stderrs

    p_values = [2 * (1 - stats.t.cdf(np.abs(t), df)) for t in t_stats]

    return coefficients, stderrs, t_stats, p_values

# no in arm

std_all = []

for home, dirs, files in os.walk(key_path):
    dirs.sort()
    for dir_name in dirs:
        indexes = os.path.join(key_path, dir_name)
        std_all.append(pd.read_csv(os.path.join(indexes, 'walking_speed.csv'))['walking speed'].std())

std_all = pd.DataFrame(std_all, columns=['std'])
max = std_all['std'].quantile(0.70)

num_1 = 0
num_2 = 0
num_3 = 0

stride_sort_1 = pd.DataFrame()
stride_sort_2 = pd.DataFrame()
stride_sort_3 = pd.DataFrame()

stride_mean_1 = []
stride_mean_2 = []
stride_mean_3 = []

corr_line_1 = []
corr_line_2 = []
corr_line_3 = []

pearson_corr_1 = []
pearson_corr_2 = []
pearson_corr_3 = []
pearson_p_1 = []
pearson_p_2 = []
pearson_p_3 = []

pearson_in_corr_1 = []
pearson_in_corr_2 = []
pearson_in_corr_3 = []
pearson_in_p_1 = []
pearson_in_p_2 = []
pearson_in_p_3 = []
no_list = ['a_092306','a_080201','a_081002', 'b_092410','b_100106','a_092506','a_080212']

for home, dirs, files in os.walk(key_path):
    dirs.sort()
    for dir_name in dirs:
        if dir_name[:-3] in no_list:
            continue
        indexes = os.path.join(key_path, dir_name)
        data_ = pd.read_csv(os.path.join(indexes, 'walking_speed.csv'))
        #data_['arm'][data_['arm'] > 1] = 1 / data_['arm'][data_['arm'] > 1]

        original_data = data_
        original_variable = original_data['walking speed']

        #normalization
        data_['walking speed'] = (data_['walking speed'] - data_['walking speed'].min()) / (data_['walking speed'].max() - data_['walking speed'].min())


        data = data_

        new_pupil_path = os.path.join(pupil_path,dir_name)
        new_key_path = os.path.join(key_path,dir_name)

        pupil_ = pd.read_csv(os.path.join(new_pupil_path, 'new_diameter_.csv'))
        mean = pupil_['rate'].mean()
        std = pupil_['rate'].std()
        pupil = pupil_[(pupil_['rate'] < mean + 3 * std) & ((pupil_['rate'] > mean - 3 * std))]


        pupil = pupil_[(pupil_['rate']<1)&((pupil_['rate']>-1))]
        pupil["rate_rolling"] = pupil['rate'].rolling(window=180).mean()


        # data
        data['stride_rolling'] = data['walking speed'].rolling(window=200).mean()
        data['converttime'] = data['time'].apply(lambda x:convert_video(x))

        original_data['stride_rolling'] = original_data['walking speed'].rolling(window=200).mean()
        original_data['converttime'] = original_data['time'].apply(lambda x: convert_video(x))



        merge = pd.merge_asof(data,pupil,left_on='converttime', right_on="time",direction="nearest")  # backward
        original_merge = pd.merge_asof(original_data,pupil,left_on='converttime', right_on="time",direction="nearest")

        if dir_name[-1] == '1' :
            stride_sort_1 = pd.concat([stride_sort_1, merge[['rate_rolling','stride_rolling']]], ignore_index=True)

            stride_mean_1.append(original_variable.mean())

            num_1 = num_1 + 1
            original_merge = original_merge.sort_values(by='rate_rolling')
            original_merge = original_merge.groupby('rate_rolling')['stride_rolling'].mean().reset_index()
            original_merge = original_merge.dropna()

            if len(original_merge) <=1: continue
            correlation_coefficient, p_value = spearmanr(original_merge['rate_rolling'],
                                                        original_merge['stride_rolling'])
            pearson_corr_1.append(correlation_coefficient)
            pearson_p_1.append(p_value)

            original_merge = original_merge[
                (original_merge['rate_rolling'] >= -0.08957766246476156) & (original_merge['rate_rolling'] <= -0.005833185017907219)]
            if len(original_merge) <= 1: continue
            correlation_coefficient, p_value = spearmanr(original_merge['rate_rolling'],
                                                        original_merge['stride_rolling'])
            pearson_in_corr_1.append(correlation_coefficient)
            pearson_in_p_1.append(p_value)

            original_merge = original_merge.iloc[::2]

            coeffs = np.polyfit(original_merge['rate_rolling'], original_merge['stride_rolling'], 1)

            original_merge['trend'] = coeffs[0] * original_merge['rate_rolling'] + coeffs[1]
            original_merge['label'] = 'person' + str(num_1)
            corr_line_1.append(original_merge[['rate_rolling', 'trend', 'label']])



        elif dir_name[0] == 'a' :
            stride_mean_2.append(original_variable.mean())

            num_2 = num_2 + 1
            stride_sort_2 = pd.concat([stride_sort_2, merge[['rate_rolling', 'stride_rolling']]], ignore_index=True)

            # merge = merge.sort_values(by='rate_rolling')
            # #merge = merge[(merge['rate_rolling'] >= -0.013) & (merge['rate_rolling'] <= 0.0783)]
            # merge = merge.groupby('rate_rolling')['stride_rolling'].mean().reset_index()
            # #merge = merge[(merge['rate_rolling'] >= -0.067) & (merge['rate_rolling'] <= 0.0092)]
            # merge = merge.dropna()

            original_merge = original_merge.sort_values(by='rate_rolling')
            original_merge = original_merge.groupby('rate_rolling')['stride_rolling'].mean().reset_index()
            original_merge = original_merge.dropna()


            if len(original_merge) <= 1: continue
            correlation_coefficient, p_value = spearmanr(original_merge['rate_rolling'], original_merge['stride_rolling'])
            pearson_corr_2.append(correlation_coefficient)
            pearson_p_2.append(p_value)

            original_merge = original_merge[(original_merge['rate_rolling'] >= -0.005833185017907219) & (
                        original_merge['rate_rolling'] <= 0.044140670424346704)]
            if len(original_merge) <= 1: continue
            correlation_coefficient, p_value = spearmanr(original_merge['rate_rolling'], original_merge['stride_rolling'])
            pearson_in_corr_2.append(correlation_coefficient)
            pearson_in_p_2.append(p_value)

            original_merge = original_merge.iloc[::2]

            coeffs = np.polyfit(original_merge['rate_rolling'], original_merge['stride_rolling'], 1)

            original_merge['trend'] = coeffs[0] * original_merge['rate_rolling'] + coeffs[1]
            original_merge['label'] = 'person' + str(num_2)
            corr_line_2.append(original_merge[['rate_rolling', 'trend', 'label']])


        elif dir_name[0] == 'b':
            stride_mean_3.append(original_variable.mean())

            num_3 = num_3 + 1
            stride_sort_3 = pd.concat([stride_sort_3, merge[['rate_rolling', 'stride_rolling']]], ignore_index=True)

            original_merge = original_merge.sort_values(by='rate_rolling')
            #merge = merge[(merge['rate_rolling'] >= 0.0783) & (merge['rate_rolling'] <= 0.151)]
            original_merge = original_merge.groupby('rate_rolling')['stride_rolling'].mean().reset_index()
            #merge = merge[(merge['rate_rolling'] >= 0.0092) & (merge['rate_rolling'] <= 0.15)]
            original_merge = original_merge.dropna()
            if len(original_merge) <= 1: continue
            correlation_coefficient, p_value = spearmanr(original_merge['rate_rolling'], original_merge['stride_rolling'])
            pearson_corr_3.append(correlation_coefficient)
            pearson_p_3.append(p_value)

            original_merge = original_merge[(original_merge['rate_rolling'] >= 0.044140670424346704) & (
                        original_merge['rate_rolling'] <= 0.11199983686890083)]

            if len(original_merge) <= 1: continue
            correlation_coefficient, p_value = spearmanr(original_merge['rate_rolling'], original_merge['stride_rolling'])
            pearson_in_corr_3.append(correlation_coefficient)
            pearson_in_p_3.append(p_value)

            original_merge = original_merge.iloc[::2]
            coeffs = np.polyfit(original_merge['rate_rolling'], original_merge['stride_rolling'], 1)

            original_merge['trend'] = coeffs[0] * original_merge['rate_rolling'] + coeffs[1]

            original_merge['label'] = 'person' + str(num_3)
            corr_line_3.append(original_merge[['rate_rolling', 'trend', 'label']])





### correlation in all
#stride_sort_1 = stride_sort_1.iloc[::20]
#stride_sort_2 = stride_sort_2.iloc[::20]
#stride_sort_3 = stride_sort_3.iloc[::20]

stride_sort_1 = stride_sort_1.sort_values(by='rate_rolling')
stride_sort_2 = stride_sort_2.sort_values(by='rate_rolling')
stride_sort_3 = stride_sort_3.sort_values(by='rate_rolling')

stride_sort_1 = stride_sort_1.groupby('rate_rolling')['stride_rolling'].mean().reset_index()
stride_sort_2 = stride_sort_2.groupby('rate_rolling')['stride_rolling'].mean().reset_index()
stride_sort_3 = stride_sort_3.groupby('rate_rolling')['stride_rolling'].mean().reset_index()

stride_sort_1 = stride_sort_1.dropna()
stride_sort_2 = stride_sort_2.dropna()
stride_sort_3 = stride_sort_3.dropna()

after_1 = stride_sort_1[ (stride_sort_1['rate_rolling'] >= -0.08957766246476156) & (stride_sort_1['rate_rolling'] <= -0.005833185017907219)]
after_2 = stride_sort_2[(stride_sort_2['rate_rolling'] > -0.005833185017907219) & (stride_sort_2['rate_rolling'] <= 0.044140670424346704)]
after_3 = stride_sort_3[(stride_sort_3['rate_rolling'] > 0.044140670424346704) & (stride_sort_3['rate_rolling'] <= 0.11199983686890083)]


def compute_confidence_interval(x, y, slope, intercept, confidence=0.95):
    y_pred = slope * x + intercept
    residuals = y - y_pred
    dof = len(y) - 2
    residual_std_error = np.sqrt(np.sum(residuals**2) / dof)
    t_value = 1.96  # For 95% confidence interval
    ci = t_value * residual_std_error * np.sqrt(1/len(x) + (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    return y_pred - ci, y_pred + ci


def format_equation(intercept, slope):
    slope_sign = "+" if slope >= 0 else "-"
    return f"y = {intercept:.2f}{slope_sign}{abs(slope):.2f}*x"

slope, intercept = np.polyfit(after_1['rate_rolling'], after_1['stride_rolling'], 1)
after_1['trend'] = slope * after_1['rate_rolling']+ intercept
lower_bound_1, upper_bound_1 = compute_confidence_interval(after_1['rate_rolling'], after_1['stride_rolling'], slope, intercept)
equation1 = format_equation(intercept, slope)

slope, intercept = np.polyfit(after_2['rate_rolling'], after_2['stride_rolling'], 1)
after_2['trend'] = slope * after_2['rate_rolling']+ intercept
lower_bound_2, upper_bound_2 = compute_confidence_interval(after_2['rate_rolling'], after_2['stride_rolling'], slope, intercept)
equation2 = format_equation(intercept, slope)

slope, intercept = np.polyfit(after_3['rate_rolling'], after_3['stride_rolling'], 1)
after_3['trend'] = slope * after_3['rate_rolling']+ intercept
lower_bound_3, upper_bound_3 = compute_confidence_interval(after_3['rate_rolling'], after_3['stride_rolling'], slope, intercept)
equation3 = format_equation(intercept, slope)

slope, intercept = np.polyfit(stride_sort_1['rate_rolling'], stride_sort_1['stride_rolling'], 1)
stride_sort_1['trend'] = slope * stride_sort_1['rate_rolling']+ intercept

slope, intercept = np.polyfit(stride_sort_2['rate_rolling'], stride_sort_2['stride_rolling'], 1)
stride_sort_2['trend'] = slope * stride_sort_2['rate_rolling']+ intercept

slope, intercept = np.polyfit(stride_sort_3['rate_rolling'], stride_sort_3['stride_rolling'], 1)
stride_sort_3['trend'] = slope * stride_sort_3['rate_rolling']+ intercept


all_lower_bounds = np.concatenate([lower_bound_1, lower_bound_2, lower_bound_3])
all_upper_bounds = np.concatenate([upper_bound_1, upper_bound_2, upper_bound_3])

y_min = np.min(all_lower_bounds)
y_max = np.max(all_upper_bounds)
#y_min, y_max = min(lower_bound_1 + lower_bound_2 + lower_bound_3), max(upper_bound_1 + upper_bound_2 + upper_bound_3)
y_range = y_max - y_min
y_axis_min = y_min - y_range/3
y_axis_max = y_max + y_range/3

plt.figure(figsize=(12, 6))


plt.plot(after_1['rate_rolling'], after_1['trend'], color='#BDD088', label="low cognitive load", \
         linewidth=3, linestyle='-')
plt.plot(after_2['rate_rolling'], after_2['trend'], color='#BDD088', label="moderate cogntive load", \
         linewidth=3, linestyle='--')
plt.plot(after_3['rate_rolling'], after_3['trend'], color='#BDD088', label="high cognitive load", \
         linewidth=3, linestyle='-.')
plt.fill_between(after_1['rate_rolling'], lower_bound_1, upper_bound_1, color='#BDD088', alpha=0.2)
plt.fill_between(after_2['rate_rolling'], lower_bound_2, upper_bound_2, color='#BDD088', alpha=0.2)
plt.fill_between(after_3['rate_rolling'], lower_bound_3, upper_bound_3, color='#BDD088', alpha=0.2)

plt.text(-0.075, y_axis_max - 0.01, 'r=-0.069, p<0.001*', fontsize=14)
plt.text(0, y_axis_max - 0.01, 'r=0.120, p<0.001*', fontsize=14)
plt.text(0.06, y_axis_max - 0.01, 'r=-0.053, p=0.001*', fontsize=14)

plt.text(-0.075, y_axis_max - 0.02, equation1, fontsize=14)
plt.text(0, y_axis_max - 0.02, equation2, fontsize=14)
plt.text(0.06, y_axis_max - 0.02, equation3, fontsize=14)


plt.xlabel('Pupil dilation ratio', fontsize = 20,fontname='Arial')
plt.ylabel('Walking speed', fontsize = 20,fontname='Arial')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False,fontsize=14)


plt.ylim(y_axis_min, y_axis_max)


def format_func(value, tick_number):
    return f"{value:.2f}"

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)




ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
output_file_path = os.path.join('G:/figure_nomal/figure-filter', 'pupil_waking_speed.png')

plt.savefig(output_file_path, dpi=300, format='png', bbox_inches='tight')
plt.show()

stride_sort_1.to_csv(os.path.join('G:/openpose_output', 'walking_speed_merge1.csv'),index=False)
stride_sort_2.to_csv(os.path.join('G:/openpose_output', 'walking_speed_merge2.csv'),index=False)
stride_sort_3.to_csv(os.path.join('G:/openpose_output', 'walking_speed_merge3.csv'),index=False)


## correlation
print('individual correlation')
correlation_coefficient, p_value = spearmanr(stride_sort_1['rate_rolling'], stride_sort_1['stride_rolling'])
print(correlation_coefficient, p_value)
correlation_coefficient, p_value = spearmanr(stride_sort_2['rate_rolling'], stride_sort_2['stride_rolling'])
print(correlation_coefficient, p_value)
correlation_coefficient, p_value = spearmanr(stride_sort_3['rate_rolling'], stride_sort_3['stride_rolling'])
print(correlation_coefficient, p_value)

print('individual coefficients')
coefficients, stderrs, t_stats, p_values = calculate_p_values(stride_sort_1['rate_rolling'], stride_sort_1['stride_rolling'])
print(coefficients, stderrs, t_stats, p_values)
coefficients, stderrs, t_stats, p_values = calculate_p_values(stride_sort_2['rate_rolling'], stride_sort_2['stride_rolling'])
print(coefficients, stderrs, t_stats, p_values)
coefficients, stderrs, t_stats, p_values = calculate_p_values(stride_sort_3['rate_rolling'], stride_sort_3['stride_rolling'])
print(coefficients, stderrs, t_stats, p_values)

print()

print('overall correlation')
correlation_coefficient, p_value = spearmanr(after_1['rate_rolling'], after_1['stride_rolling'])
print(correlation_coefficient, p_value)
correlation_coefficient, p_value = spearmanr(after_2['rate_rolling'], after_2['stride_rolling'])
print(correlation_coefficient, p_value)
correlation_coefficient, p_value = spearmanr(after_3['rate_rolling'], after_3['stride_rolling'])
print(correlation_coefficient, p_value)

print('overall coefficients')
coefficients, stderrs, t_stats, p_values = calculate_p_values(after_1['rate_rolling'], after_1['stride_rolling'])
print(coefficients, stderrs, t_stats, p_values)
coefficients, stderrs, t_stats, p_values = calculate_p_values(after_2['rate_rolling'], after_2['stride_rolling'])
print(coefficients, stderrs, t_stats, p_values)
coefficients, stderrs, t_stats, p_values = calculate_p_values(after_3['rate_rolling'], after_3['stride_rolling'])
print(coefficients, stderrs, t_stats, p_values)

# print(sum(pearson_corr_2)/len(pearson_corr_2))
# print(sum(pearson_p_2)/len(pearson_p_2))
#
# print(sum(pearson_corr_3)/len(pearson_corr_3))
# print(sum(pearson_p_3)/len(pearson_p_3))



