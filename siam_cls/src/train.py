import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from train_epoch import train_epoch, test_epoch, train_epoch_triplet, test_epoch_triplet
from models import SiameseNetwork, TripletNetwork
from dataset import get_train_val_data
from config import Config
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import neptune.new as neptune
import torch.nn as nn


def train_triplets(config, neptune_run):
    config = Config()
    train_loader, test_loader = get_train_val_data(config)
    model = TripletNetwork()
    device = config.device
    model = torch.jit.script(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)
    for epoch in tqdm(range(config.num_epochs)):
        train_epoch_triplet(model, device, train_loader, optimizer, epoch)
        test_epoch_triplet(model, device, train_loader, test_loader)   
        scheduler.step() 

        
def train_siamese(config, neptune_run):
    train_loader, test_loader = get_train_val_data(config)
    model = SiameseNetwork()
    device = config.device
    model = torch.jit.script(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_metric = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)
    criterion = nn.BCELoss()
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, metrics = test_epoch(model, device, test_loader, criterion)  
        print('\nEpoch: {}, Train_loss: {:.4f}, Test set: Test_loss: {:.4f}, Metrics: {:.3f} \n'
            .format(epoch, train_loss, test_loss, metrics.mean()))
        neptune_run['/'.join(['train', 'train_loss'])].log(round(train_loss,4))
        neptune_run['/'.join(['test', 'test_loss'])].log(round(test_loss,4))
        for cls_name, val in list(zip(['different','equal'],metrics)):
            neptune_run['/'.join(['test', cls_name, 'f1_score'])].log(round(val,3))
        if np.mean(metrics)>=best_metric:
            best_metric = np.mean(metrics)
            #torch.save(model.state_dict(), 'src/weights/best_all_combs_test.pth')
        scheduler.step(test_loss) 
        
if __name__ == "__main__":
    config = Config()
    
    neptune_run = neptune.init_run(
    project=config.neptune_project,
    api_token=config.neptune_api_token
    )
    
    train_siamese(config, neptune_run)
