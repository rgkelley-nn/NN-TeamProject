import model_update
import torch
import os
import glob
from model_update import LSTMForecaster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMForecaster(input_dim=1, hidden_dim=64, dropout=0.1).to(device)

path = "data/energyData/validation/"
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
os.chdir("..")
os.chdir("..")


model.eval()
with torch.no_grad():
    val_pred = model(Xva)
    val_loss = loss_fn(val_pred, yva).item()