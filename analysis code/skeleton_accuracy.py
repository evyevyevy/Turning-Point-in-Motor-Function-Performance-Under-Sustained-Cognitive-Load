import numpy as np
import jsonlines
import pandas as pd
import os
import cv2
import random
import pickle



def pwrite(dirpath_json, dirname):

    accuracy = []
    with open(os.path.join(dirpath_json, dirname + '.jsonl'), "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):

            if len(item['people']) == 0 :
                #print('None')
                continue
            accuracy_1 = item['people'][0]['pose_keypoints'][5]
            if accuracy_1 < 0.7 :

                continue
            else:
                accuracy.append(accuracy_1)



    print(dirname + ' finish')
    return sum(accuracy) / len(accuracy)


json_path = 'F:/openpose_output'
mean_accuracy = []
if __name__ == '__main__':
    for home, dirs, files in os.walk(json_path):
        for dir in dirs:
            dirpath_json = os.path.join(json_path, dir)
            dirname = dir[2:]
            #if os.path.exists(os.path.join(dirpath, dirname + '.pkl')):
                #print(dirname + ' exist')
                #continue

            m = pwrite(dirpath_json, dirname)
            mean_accuracy.append(m)

    print(sum(mean_accuracy)/len(mean_accuracy))

