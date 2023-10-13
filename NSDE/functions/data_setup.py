import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
import os
NUM_WORKERS = os.cpu_count()

def create_xy(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return torch.tensor(dataX).to(torch.float32), torch.tensor(dataY).to(torch.float32)

def create_dataloaders(
    train_data,
    test_data,
    batch_size: int,
    num_workers: int=NUM_WORKERS):


  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      num_workers=num_workers,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      num_workers=num_workers,
  )

  return train_dataloader, test_dataloader
