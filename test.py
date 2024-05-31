from model import *
import pandas as pd
from sklearn.model_selection import train_test_split

print("Reading...")
df = pd.read_csv("/home/ids/NT505.O21/Datasets/Changed_dataset/CICIDS2017-FullHD.csv")
# df = pd.DataFrame(data)
print("Successfully read data.")

X = df.drop(columns=["Label"])
y = df['Label']
print(X.shape)
print(y.shape)

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
print("Successfully split.")

dnn = DFNN(78, 15)
predict = dnn.predict(X)
print(len(predict))