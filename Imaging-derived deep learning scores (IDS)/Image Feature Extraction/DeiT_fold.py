#!/usr/bin/env python
# coding: utf-8

# In[11]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
# Accept two command-line arguments: job_index and fold_name
job_index = sys.argv[1]
fold_name = sys.argv[2]
print("B_scan : ",job_index, flush=True)

import os
import gc
import pandas as pd
from PIL import Image
from torchvision import transforms
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

label_file = '/storage0/lab/khm1576/Workspace/disease/Glaucoma_dat.txt'

# Read the labels CSV
df = pd.read_csv(label_file, sep="\t")
df = df.dropna(subset=['Gla'])
df['app77890'] = df['app77890'].astype(str).str.replace('.0', '', regex=False)
id_label_map = dict(zip(df['app77890'], df['Gla']))

id_file = '/storage0/lab/khm1576/IDPs/OCT/OCT_id.txt'
df_id = pd.read_csv(id_file, sep="\t", header=None, usecols=[0])

filtered_df = df[df['app14048'].isin(df_id[0])]
id_label_map = filtered_df.set_index('app77890')['Gla'].to_dict()

fold = '/storage0/lab/khm1576/IDPs/OCT/{fold_name}.csv'
fold_id = pd.read_csv(fold, sep="\t", header=None)
fold_id[0] = fold_id[0].astype(str)

left_existing_ids = pd.read_csv('/storage0/lab/khm1576/Image/OCT/left_existing_ids.csv', sep="\t", header=None)
left_existing_ids[0] = left_existing_ids[0].astype(str)

right_existing_ids = pd.read_csv('/storage0/lab/khm1576/Image/OCT/right_existing_ids.csv', sep="\t", header=None)
right_existing_ids[0] = right_existing_ids[0].astype(str)



train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(20),     
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])
train_right_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=1.0), 
    transforms.RandomRotation(20),      
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])




valid_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
valid_test_right_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=1.0), 
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

right_train_dataset = OCTDataset(id_label_map, right_train_ids, right_image_folder, transform=train_right_transform)
right_valid_dataset = OCTDataset(id_label_map, right_valid_ids, right_image_folder, transform=valid_test_right_transform)

# Merge the Left and Right datasets by concatenating them
train_dataset = left_train_dataset + right_train_dataset
valid_dataset = left_valid_dataset + right_valid_dataset
# test_dataset = train_dataset + valid_dataset

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

dataloaders = {
    'train': train_loader,
    'val': valid_loader,
}

# Calculate dataset sizes
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(valid_dataset),
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


print("B_scan : ",job_index,"deit fold0")

import torch
import timm


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModifiedDeit(nn.Module):
    def __init__(self, model_name="deit_tiny_patch16_224", pretrained=True):
        super(ModifiedDeit, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

     
        if isinstance(self.model.head, nn.Linear):
            in_features = self.model.head.in_features
        else:
            in_features = self.model.head[-1].in_features
        
        self.model.head = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)



deit = timm.create_model('deit_tiny_patch16_224', pretrained=True)
deit = ModifiedDeit(pretrained=True).to(Device)


for param in deit.model.parameters():
            param.requires_grad = True



optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, deit.parameters()), eps=1e-8, lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3)

import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score

import time
import copy
import torch
from sklearn.metrics import roc_auc_score

def train_deit(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=50, early_stopping_patience=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 10)

        total_num_true_1_train = 0
        total_num_pred_1_train = 0
        total_num_true_1_val = 0
        total_num_pred_1_val = 0
   
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

        
            for inputs, labels, person_id in dataloaders[phase]:
                inputs, labels = inputs.to(Device), labels.to(Device)
                labels = labels.view(-1,1)  

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
          
                    assert outputs.shape == labels.shape, f"Shape mismatch! Outputs: {outputs.shape}, Labels: {labels.shape}"
                    
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

      
                running_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.4).float()  # 🔹 0.5 → 0.4로 조정하여 양성 판별 민감도 증가

                running_corrects += torch.sum(preds == labels.data)
                num_pred_1 = torch.sum(preds == 1).item()
                num_true_1 = torch.sum(labels == 1).item()

                total_num_true_1 += num_true_1
                total_num_pred_1 += num_pred_1

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().detach().numpy())
                all_probs.extend(probs.cpu().detach().numpy())

            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_auc = roc_auc_score(all_labels, all_probs)
            print(f"Number of predictions as 1: {total_num_pred_1}")
            print(f"Number of true labels as 1: {total_num_true_1}")

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')
        
       
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

  
                scheduler.step(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    model.load_state_dict(best_model_wts)
    return model


criterion = FocalLoss(alpha=3, gamma=2, reduction='mean')
deit = train_deit(deit, criterion, optimizer_ft, scheduler, dataloaders, dataset_sizes, num_epochs=50, early_stopping_patience=5)


torch.save(deit, f'/home/guestuser1/deit/fold/{fold_name}/{fold_name}_model/deit_{job_index}.pt')
def evaluate(model, test_loader):
    model.eval()  
    all_labels = []
    all_preds = []
    all_ids = []

    with torch.no_grad(): 
        for data, target, person_id in test_loader:  
            data, target = data.to(Device), target.to(Device)  
            output = model(data)
            
  
            output = torch.sigmoid(output).cpu().numpy() 
            target = target.cpu().numpy() 
            

            all_preds.extend(output.flatten())  
            all_labels.extend(target.flatten()) 
            all_ids.extend(person_id)  

   
    auc = roc_auc_score(all_labels, all_preds)
    

    pred = pd.DataFrame({
        'ID': all_ids,
        'Actual': all_labels,
        'Predicted_Prob': all_preds
    })
    

    return pred, auc


pred, auc_score = evaluate(deit, valid_loader)
print(f'deit fold AUC: {auc_score:.4f}')



pred.to_csv(f'/home/guestuser1/deit/fold/fold0/fold0_pred/{job_index}_pred.csv', index=False)


torch.cuda.empty_cache()


del train_loader, valid_loader, train_dataset, valid_dataset  
gc.collect()

time.sleep(120)

os._exit(0)




# In[ ]:




