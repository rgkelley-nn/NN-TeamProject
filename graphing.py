import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

path = "lossData/"
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
os.chdir("..")

data = []
for i in result:
    data.append(pd.read_csv(path+i))

for idx, i in enumerate(data):
    plt.plot(i["val_mse"], label=f"{result[idx][:-4]}")
plt.legend()
plt.show()