import os
import pandas as pd
import numpy as np


def split_dataset(df):
    BENIGN_LABEL = df[df['Label'] == 'BENIGN']
    ATTACK_LABEL = df[df['Label'] != 'BENIGN']

    random_attack_labels = [ 0, 3, 5, 1, 2, 6, 4, 14, 13]
    AttackandBenign = ATTACK_LABEL[ATTACK_LABEL['Label'].isin(random_attack_labels)]
    AttackRemain = ATTACK_LABEL[~ATTACK_LABEL['Label'].isin(random_attack_labels)]

    df_benign = pd.concat([BENIGN_LABEL, AttackandBenign], ignore_index=True)
    df_attack = pd.concat([AttackRemain], ignore_index=True)
    return df_benign, df_attack


file_path = '/home/ids/NT505.O21/Datasets/Changed_dataset/CICIDS2017-FullHD.csv'
df = pd.read_csv(file_path)
df_benign, df_attack = split_dataset(df)

path = '/home/ids/NT505.O21/Datasets/Changed_dataset/'
df_benign.to_csv(os.path.join(path, 'CICIDS2017-9LABEL.csv'), index=False)
df_attack.to_csv(os.path.join(path, 'CICIDS2017-6LABEL.csv'), index=False)

num_sample_9 = len(df_benign)
print(num_sample_9)
num_sample_6 = len(df_attack)
print(num_sample_6)
print(df_benign['Label'].unique()) #old_class
print(df_attack['Label'].unique()) #new_class

labels9 = "/home/ids/NT505.O21/Datasets/Changed_dataset/CICIDS2017-9LABEL.csv"
labels6 = "/home/ids/NT505.O21/Datasets/Changed_dataset/CICIDS2017-6LABEL.csv"

df_benign = pd.read_csv(labels9)
print(df_benign["Label"].unique())