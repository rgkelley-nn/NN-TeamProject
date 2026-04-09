import pandas as pd
import numpy as np
import models
import torch
import matplotlib.pyplot as plt


df = pd.read_csv('data/AEP_hourly.csv')

dataLen = len(df)

kSplits = 8
kLen = dataLen/kSplits

# print(type(df[(1 * dataLen):((1+1) * dataLen +1)]))

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)\

seq_length = 10
# X, y = create_sequences(data, seq_length)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.LSTMModel(1, 64, 64, 1).to(device)
criteria = models.getLoss(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

h0, c0 = None, None
lossarr = []
for i in range(kSplits - 1):
    
    model.train()
    trainData = df[(i * dataLen):((i+1) * dataLen +1)]
    tensorData = torch.tensor(trainData.AEP_MW[:,:])
    out, h0, c0 = model(tensorData, h0, c0)
    loss = criteria(tensorData, out)
    lossarr.append(loss)

plt.plot(lossarr)
    

    


