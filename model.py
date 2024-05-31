import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

class MyDataset(Dataset):
    def __init__(self, name, dataframe: pd.DataFrame, label_column):
        self.dataset = dataframe
        self.label_column = label_column
        self.name = name
        self._bookkeeping_path = os.path.join("/home/ids/NT505.O21/Datasets/cached_datasets/", f"{name}.pkl")

    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        label = torch.tensor(row[self.label_column], dtype=torch.long)
        data = torch.tensor(row.drop(self.label_column), dtype=torch.float)
        return data, label
    
    def __len__(self):
        return len(self.dataset)

class DFNN(nn.Module):
    def __init__(self, input_shape: int, num_classes: int):
        super(DFNN, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.__init_model__()
        # self.double()

    def __init_model__(self):
        model = nn.Sequential(
            nn.Linear(self.input_shape, 120),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(120, self.input_shape),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.input_shape, self.num_classes)
        )
        return model
    
    def forward(self, X):
        logits = self.model(X)
        return logits
    
    # def fit(self, X, y, batch_size, epochs, verbose=False):
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(self.parameters())

    #     for epoch in range(epochs):
    #         running_loss = 0.0
    #         for i in range(0, len(X), batch_size):
    #             inputs = torch.tensor(X[i:i+batch_size].values, dtype=torch.float)
    #             labels = torch.tensor(y[i:i+batch_size].values, dtype=torch.float)

    #             optimizer.zero_grad()
    #             outputs = self(inputs)
                
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()

    #             running_loss += loss.item()

    #         if verbose:
    #             print(f"Epoch {epoch+1}, Loss: {running_loss}")

    # def predict(self, X):
    #     inputs = torch.tensor(X.values, dtype=torch.float32)
    #     outputs = self(inputs)
    #     _, predicted = torch.max(outputs, 1)
    #     print("Logit value: {}".format(predicted))
    #     return predicted.numpy()
    