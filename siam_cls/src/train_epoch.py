import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score


def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for (images_1, images_2, targets) in tqdm(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        targets = targets.to(torch.float32)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        train_loss += loss.sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader.dataset)
    return train_loss

def train_epoch_triplet(model, device, train_loader, optimizer):
    model.train()

    criterion = nn.TripletMarginLoss(margin=0.1, p=0.5)

    for _, (images_anchor, images_pos, images_neg, targets) in enumerate(train_loader):
        images_anchor, images_pos, images_neg, targets = images_anchor.to(device), images_pos.to(device), images_neg.to(device), targets.to(device)
        targets = targets.to(torch.float32)
        optimizer.zero_grad()
        outputs_anchor, outputs_pos, outputs_neg = model(images_anchor, images_pos, images_neg)
        loss = criterion(outputs_anchor, outputs_pos, outputs_neg)
        loss.backward()
        optimizer.step()


def test_epoch(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    targets_labels, predicts_labels = [], []

    with torch.no_grad():
        for (images_1, images_2, targets) in tqdm(test_loader):
            for target in targets.numpy():
                targets_labels.append(target)
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            targets = targets.to(torch.float32)
            outputs = model(images_1, images_2).squeeze()
            print()
            print("PRE = ",outputs)
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            outputs = torch.where(outputs > 0.5, 1, 0).cpu().numpy() # get the index of the max log-probability
            print()
            print("POST = ",outputs)
            print()
            for pred in outputs:
                predicts_labels.append(pred)

    test_loss /= len(test_loader.dataset)
    targets_labels, predicts_labels = np.array(targets_labels), np.array(predicts_labels)
    metrics = np.array(f1_score(targets_labels.astype(int), predicts_labels.astype(int), average=None))
    return test_loss, metrics
    
def collect_embedings(train_loader, model, device):
    embeds = []  
    labels = []
    with torch.no_grad():
        for img, _, _, label in tqdm(train_loader):
            embeds.append(model.forward_once(img.to(device)).cpu().numpy())
            labels.append(label)
        
    embeds = np.concatenate(embeds)
    labels = np.concatenate(labels)
    return embeds, labels

    
def test_epoch_triplet(model, device, train_loader, test_loader):
    model.eval()
    embeds, embeds_labels = collect_embedings(train_loader, model, device)
    targets = []
    test_embeds = []
    with torch.no_grad():
        for (images, _, _, target_label) in test_loader:
            test_embeds.append(model.forward_once(images.to(device)).cpu().numpy())
            targets.append(target_label)
    test_embeds = np.concatenate(test_embeds)
    targets = np.concatenate(targets)
    cls = XGBClassifier()
    cls.fit(embeds, embeds_labels)     
    predicts = cls.predict(test_embeds)
    print("accuracy_score = ",accuracy_score(targets, predicts))
    
     
    
