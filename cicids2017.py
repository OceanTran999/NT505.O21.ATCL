import os
from model import DFNN
from model import MyDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def combine(src_dir: str) -> pd.DataFrame:
    # Remove unknown character from 'Labels' column
    print("Remove unknown character.")
    file_path = os.path.join(src_dir, "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    labels = {"Web Attack � Brute Force": "Web Attack-Brute Force",
              "Web Attack � XSS": "Web Attack-XSS",
              "Web Attack � Sql Injection": "Web Attack-Sql Injection"}
    df = pd.read_csv(file_path, skipinitialspace=True)
    for old_label, new_label in labels.items():
        df.Label.replace(old_label, new_label, inplace=True)
    df.to_csv(file_path, index=False)
    
    # Combine files
    print("Combine files.")
    file_names = ["Monday-WorkingHours.pcap_ISCX.csv",
                  "Tuesday-WorkingHours.pcap_ISCX.csv",
                  "Wednesday-workingHours.pcap_ISCX.csv",
                  "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                  "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                  "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                  "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                  "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
    df = [pd.read_csv(os.path.join(src_dir, f), skipinitialspace=True) for f in file_names]
    df = pd.concat(df, ignore_index=True)
    return df

def preprocessing(df: pd.DataFrame, col_label: str = "Label") -> pd.DataFrame:
    # Simplify the class label
    print("Simplify the class label.")
    labels = {"Label": {"BENIGN": 0,
                "DoS Hulk": 1,
                "PortScan": 2,
                "DDoS": 3,
                "DoS GoldenEye": 4,
                "FTP-Patator": 5,
                "SSH-Patator": 6,
                "DoS slowloris": 7,
                "DoS Slowhttptest": 8,
                "Bot": 9,
                "Web Attack-Brute Force": 10,
                "Web Attack-XSS": 11,
                "Infiltration": 12,
                "Web Attack-Sql Injection": 13,
                "Heartbleed": 14}}
    for k, v in labels["Label"].items():
        print("{:>2} - {}".format(v, k))
    df.replace(labels, inplace=True)
    
    # Remove duplicate column
    print("Remove duplicate columns.")
    df = df.drop(columns=['Fwd Header Length.1'])
    
    #  Fill NaN with median value of each class
    print("Fill NaN in {} rows with median value of each class.".format(df.isna().any(axis=1).sum()))
    y = df[[col_label]]
    df = df.groupby(col_label).transform(lambda x: x.fillna(x.median()))
    
    # Replace infinite values with twice of maximum value of each class.
    df.replace([np.inf], np.nan, inplace=True)
    print("Replace {} Inf values with twice of maximum value of each class.".format(df.isna().sum().sum()))
    df = pd.concat([df, y], axis=1, sort=False)
    df = df.groupby(col_label).transform(lambda x: x.fillna(x.max() * 2))
    
    # Merge
    df = pd.concat([df, y], axis=1, sort=False)
    X = df.drop(columns=[col_label])
    y = df[col_label]
    return df

def balance_data(X, y):
    sampling_strategy = {0: 880771}
    undersample = NearMiss(sampling_strategy=sampling_strategy)
    pipeline = Pipeline(steps=[('u', undersample)])
    X_resample, y_resample = pipeline.fit_resample(X, y)
    return X_resample, y_resample



path = '/home/ids/NT505.O21/Datasets/CIC-IDS2017'
new_path = '/home/ids/NT505.O21/Datasets/Changed_dataset'

print("Combine datasets....")
dataset = combine(path)

print("Data preprocessing...")
dataset = preprocessing(dataset)

X = dataset.drop(columns=["Label"])
y = dataset["Label"]

X_balanced, y_balanced = balance_data(X, y)
count_y = Counter(y_balanced)
print(count_y)
print("Sampling Label")

print("Data")

print("Saving dataset....")
dataset = dataset.sample(frac=1)
dataset.to_csv(os.path.join(new_path, "CICIDS2017-FullHD.csv"), index= True)

num_sample_full = len(dataset)
print(num_sample_full)
print(dataset['Label'].unique())

