import os
import pandas as pd
import numpy as np
from model import DFNN
from model import MyDataset
from sklearn.model_selection import train_test_split
import random



labels9 = "/home/ids/NT505.O21/Datasets/Changed_dataset/CICIDS2017-9LABEL.csv"
labels10 = "/home/ids/NT505.O21/Datasets/Changed_dataset/CICIDS2017-10LABEL.csv"
labels11 = "/home/ids/NT505.O21/Datasets/Changed_dataset/CICIDS2017-11LABEL.csv"
labels4 = "/home/ids/NT505.O21/Datasets/Changed_dataset/CICIDS2017-4LABEL.csv"
labels5 = "/home/ids/NT505.O21/Datasets/Changed_dataset/CICIDS2017-5LABEL.csv"
labels6 = "/home/ids/NT505.O21/Datasets/Changed_dataset/CICIDS2017-6LABEL.csv"

df_benign9 = pd.read_csv(labels9)
df_benign10 = pd.read_csv(labels10)
df_benign11 = pd.read_csv(labels11)
df_attack4 = pd.read_csv(labels4)
df_attack5 = pd.read_csv(labels5)
df_attack6 = pd.read_csv(labels6)

print(df_attack6['Label'].unique())
print(df_benign9['Label'].unique())

random_labels = [0,1,2,3,4,5,6,13,14]
selected_labels = random.sample(random_labels, 5)
df_8labels= df_benign9[df_benign9['Label'].isin(selected_labels)]

df_combined = pd.concat([df_attack6, df_8labels], ignore_index=True)
print(df_combined['Label'].unique())
path = '/home/ids/NT505.O21/Datasets/Changed_dataset/'
df_combined.to_csv(os.path.join(path, 'CICIDS2017-Combined_6_5.csv'), index=False)






