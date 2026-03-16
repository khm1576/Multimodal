#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
# Accept two command-line arguments: job_index and fold_name
job_index = sys.argv[1]
fold_name = sys.argv[2]
print(f"Job index: {job_index}, Fold: {fold_name}", flush=True)

import os
import gc
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import copy

from torchvision import models
from sklearn.metrics import roc_auc_score

import torch.nn.functional as F

# Label file path
label_file = '/storage0/lab/khm1576/Workspace/disease/Glaucoma_dat.txt'

# Read the labels CSV
df = pd.read_csv(label_file, sep="\t")
df = df.dropna(subset=['Gla'])
df['app'] = df['app'].astype(str).str.replace('.0', '', regex=False)
id_label_map = dict(zip(df['app'], df['Gla']))

id_file = '/storage0/lab/khm1576/IDPs/OCT/OCT_id.txt'
df_id = pd.read_csv(id_file, sep="\t", header=None, usecols=[0])

filtered_df = df[df['filter'].isin(df_id[0])]
id_label_map = filtered_df.set_index('app')['Gla'].to_dict()

# Use the provided fold name to load the fold CSV file
fold = f'/storage0/lab/khm1576/IDPs/OCT/{fold_name}.csv'
fold_id = pd.read_csv(fold, sep="\t", header=None)
fold_id[0] = fold_id[0].astype(str)

left_existing_ids = pd.read_csv('/storage0/lab/khm1576/Image/OCT/left_existing_ids.csv', sep="\t", header=None)
left_existing_ids[0] = left_existing_ids[0].astype(str)

right_existing_ids = pd.read_csv('/storage0/lab/khm1576/Image/OCT/right_existing_ids.csv', sep="\t", header=None)
right_existing_ids[0] = right_existing_ids[0].astype(str)

# Training image augmentation transforms
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),  # Horizontal flip
    transforms.RandomVerticalFlip(),    # Vertical flip
    transforms.RandomRotation(20),      # Random rotation up to 20 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Random translation up to 10%
    transforms.ToTensor()
])

# Validation/Test transforms (without augmentation)
valid_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# Custom dataset class for both Left and Right images
class OCTDataset(Dataset):
    def __init__(self, id_label_map, ids, image_folder, transform=None):
        self.id_label_map = {id: id_label_map[id] for id in ids}  # Filtered by existing images
        self.ids = list(ids)
        self.image_folder = image_folder
        self.transform = transform
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        person_id = self.ids[idx]
        label = self.id_label_map[person_id]
        
        # Get the image path for the specific folder (Left or Right)
        image_path = os.path.join(self.image_folder, person_id, f"{person_id}_{job_index}.png")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Warning: Image not found for {person_id}")
        
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.float32), person_id

# Split the IDs into validation and training sets
left_valid_ids = set(fold_id[0]) & set(left_existing_ids[0])
left_train_ids = set(left_existing_ids[0]) - left_valid_ids

right_valid_ids = set(fold_id[0]) & set(right_existing_ids[0])
right_train_ids = set(right_existing_ids[0]) - right_valid_ids

# Create Dataset objects for Left and Right datasets
left_image_folder = '/storage0/lab/khm1576/Image/OCT/Left'
right_image_folder = '/storage0/lab/khm1576/Image/OCT/Right'

left_train_dataset = OCTDataset(id_label_map, left_train_ids, left_image_folder, transform=train_transform)
left_valid_dataset = OCTDataset(id_label_map, left_valid_ids, left_image_folder, transform=valid_test_transform)

right_train_dataset = OCTDataset(id_label_map, right_train_ids, right_image_folder, transform=train_transform)
right_valid_dataset = OCTDataset(id_label_map, right_valid_ids, right_image_folder, transform=valid_test_transform)

# Merge the Left and Right datasets by concatenating them
train_dataset = left_train_dataset + right_train_dataset
valid_dataset = left_valid_dataset + right_valid_dataset


# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)


dataloaders = {
    'train': train_loader,
    'val': valid_loader
}

# Calculate dataset sizes
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(valid_dataset)
}

# Check dataset sizes
print(f"Train set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Load pre-trained ResNet-18 model
resnet = models.resnet18(weights="IMAGENET1K_V1")

# Redefine layer3 and layer4 to add Dropout after layer4
class ModifiedResNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedResNet, self).__init__()
        self.original_model = original_model
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = nn.Sequential(
            original_model.layer4,
            nn.Dropout(0.5)  # Add Dropout after layer4
        )
        self.avgpool = original_model.avgpool
        num_ftrs = original_model.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout rate
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        x = self.original_model.conv1(x)
        x = self.original_model.bn1(x)
        x = self.original_model.relu(x)
        x = self.original_model.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

resnet = ModifiedResNet(resnet)

# Freeze all parameters, then enable training for specific layers
for param in resnet.parameters():
    param.requires_grad = False

# Enable training for layer4 parameters
for param in resnet.layer4.parameters():
    param.requires_grad = True

# Enable training for fc layers
for param in resnet.fc.parameters():
    param.requires_grad = True

resnet = resnet.to(DEVICE)

# Define optimizer with L2 regularization
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), eps=1e-8, lr=0.0001, weight_decay=1e-4)

# Learning rate scheduler (ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3)

from sklearn.metrics import roc_auc_score, precision_score, recall_score

def train_resnet(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, early_stopping_patience=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_auc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 10)
        
        total_num_true_1_train = 0
        total_num_pred_1_train = 0
        total_num_true_1_val = 0
        total_num_pred_1_val = 0
        
        # Training and validation phases for each epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                total_num_true_1 = total_num_true_1_train
                total_num_pred_1 = total_num_pred_1_train
                model.train()
            else:
                total_num_true_1 = total_num_true_1_val
                total_num_pred_1 = total_num_pred_1_val
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []
            all_probs = []

            # Iterate over data.
            for inputs, labels, person_id in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                labels = labels.view(-1, 1)  # Adjust shape

                optimizer.zero_grad()

                # Compute model outputs
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # Ensure output and label shapes match
                    assert outputs.shape == labels.shape, f"Shape mismatch! Outputs: {outputs.shape}, Labels: {labels.shape}"
                    
                    # Compute loss
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Accumulate loss and predictions for AUC computation
                running_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.4).float()  # Lower threshold to 0.4 for more positive predictions
                running_corrects += torch.sum(preds == labels.data)

                num_pred_1 = torch.sum(preds == 1).item()
                num_true_1 = torch.sum(labels == 1).item()

                total_num_true_1 += num_true_1
                total_num_pred_1 += num_pred_1
                
                probs = torch.sigmoid(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().detach().numpy())
                all_probs.extend(probs.cpu().detach().numpy())

            # Compute epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_auc = roc_auc_score(all_labels, all_probs)

            print(f"Number of predictions as 1: {total_num_pred_1}")
            print(f"Number of true labels as 1: {total_num_true_1}")

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')
        
            # Update the model based on validation loss
            if phase == 'val':
                if epoch > 10:
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                if epochs_no_improve == early_stopping_patience:
                    print(f'Early stopping at epoch {epoch}')
                    model.load_state_dict(best_model_wts)
                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    return model

                # Adjust learning rate based on validation loss
                scheduler.step(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    model.load_state_dict(best_model_wts)
    return model

# Use Focal Loss as the criterion
criterion = FocalLoss(alpha=3, gamma=2, reduction='mean')

model_resnet18 = train_resnet(resnet, criterion, optimizer_ft, scheduler, dataloaders, dataset_sizes, num_epochs=50, early_stopping_patience=5)

# Save the trained model using the provided fold name in the path
torch.save(model_resnet18, f'/storage0/lab/khm1576/Image/OCT/{fold_name}/{fold_name}_model/resnet18_{job_index}.pt')

def evaluate(model, test_loader):
    model.eval()  
    all_labels = []
    all_preds = []
    all_ids = []

    with torch.no_grad(): 
        for data, target, person_id in test_loader:  
            data, target = data.to(DEVICE), target.to(DEVICE)  
            output = model(data)
            
            # Convert model outputs to probabilities
            output = torch.sigmoid(output).cpu().numpy()
            target = target.cpu().numpy()
            
            # Store predictions and labels
            all_preds.extend(output.flatten())
            all_labels.extend(target.flatten())
            all_ids.extend(person_id)

    # Compute AUC
    auc = roc_auc_score(all_labels, all_preds)
    
    # Save results in a pandas DataFrame
    pred = pd.DataFrame({
        'ID': all_ids,
        'Actual': all_labels,
        'Predicted_Prob': all_preds
    })
    
    return pred, auc

pred, auc_score = evaluate(model_resnet18, test_loader)
print(f'ResNet AUC: {auc_score:.4f}')

# Save predictions using the provided fold name in the path
pred.to_csv(f'/storage0/lab/khm1576/Image/OCT/{fold_name}/{fold_name}_pred/{job_index}_pred.csv', index=False)

# Clear GPU memory
torch.cuda.empty_cache()

# Clear CPU memory by deleting data loaders and datasets
del train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset
gc.collect()

# Delay before exiting
time.sleep(120)

os._exit(0)
