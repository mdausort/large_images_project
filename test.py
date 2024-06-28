
import os
import sys
import wandb
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


cluster = True

if cluster:
    sys.path.insert(0, '/auto/home/users/m/d/mdausort/Software/large_image_source_tifffile')
else:
    sys.path.insert(0, 'C:/Users/dausort/OneDrive - UCL/Bureau/large_image_source_tifffile')
    
import large_image

class DBTADataset(Dataset):

    def __init__(self, data_dir, annotation_file, magnification, transform=None):
        self.data_dir = data_dir
        self.annotation_path = os.path.join(self.data_dir, annotation_file)
        self.images_dir = self.data_dir
        self.transform = transform
        self.magnification = magnification
        self.classes = [1, 2]

    def __len__(self):
        df = pd.read_csv(self.annotation_path)
        return len(df)
#        images = [f for f in os.listdir(self.images_dir) if f.endswith('.ndpi')]
#         
#        return len(images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        df = pd.read_csv(self.annotation_path)

        class_name = df.loc[idx, 'diagnosis']
        class_ = self.classes.index(class_name)

        image_name = df.loc[idx, 'uuid']
        xc = df.loc[idx, 'xc']
        yc = df.loc[idx, 'yc']
        w = df.loc[idx, 'w']
        h = df.loc[idx, 'h']

        image_path = os.path.join(self.images_dir, image_name)
        image_slide = large_image.getTileSource(image_path)

        # Extract the region
        slide_info = image_slide.getSingleTile(tile_size=dict(width=w, height=h, left=xc, top=yc),
                                               scale=dict(magnification=self.magnification),
                                               format=large_image.tilesource.TILE_FORMAT_NUMPY)
        image_patch = np.copy(slide_info['tile'])

        if self.transform:
            image_patch = self.transform(image_patch)

        return image_patch, class_
        

def test(patch_w=224, patch_h=224, stride_percent=1.0, magnification=20.0, model='vgg16', name_run='debug',
         bs=64, lr=0.001, comment='no_one'):

    annotation_file = 'annotation_patches_' + str(patch_w) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification)

    path_data = '/CECI/home/users/m/d/mdausort/Cytology/'
    data_dir = os.path.join(path_data, 'Training/')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create datasets and dataloaders
    test_dataset = DBTADataset(data_dir, annotation_file + '_test.csv', magnification, transform)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

    num_classes = len(test_dataset.classes)

    # Load the pre-trained model
    if model == 'vgg16':
        model_used = models.vgg16(weights='IMAGENET1K_V1')
        model_used.classifier[6] = nn.Linear(model_used.classifier[6].in_features, num_classes)
    elif model == 'resnet50':
        model_used = models.resnet50(weights='IMAGENET1K_V1')
        model_used.fc = nn.Linear(model_used.fc.in_features, num_classes)
    else:
        print('Model name is incorrect.')

    # Move the model to the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_used = model_used.to(device)

    # Load the best model weights
    model_used.load_state_dict(torch.load(path_data + 'best_epoch_weights.pth'))

    # Set model to evaluation mode
    model_used.eval()

    # Testing loop
    all_labels = []
    all_preds = []
    all_inputs = []

    test_loss = 0.0
    running_corrects = 0

    criterion = nn.CrossEntropyLoss()

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        print(inputs.shape)
        
        with torch.no_grad():
            outputs = model_used(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        test_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_inputs.extend(inputs.cpu().numpy())
    
    # print(len(all_labels))

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_inputs = np.array(all_inputs)
    
    true_pos = 0
    true_neg = 0
    false_pos = 1
    false_neg = 1
    
    for i in range(len(all_labels)):
        if all_labels[i] == 1 and all_preds[i] == 1:
            true_pos += 1
        elif all_labels[i] == 2 and all_preds[i] == 2:
            true_neg += 1
        elif all_preds[i] == 1 and all_labels[i] == 2:
            false_pos += 1
        elif all_preds[i] == 2 and all_labels[i] == 1:
            false_neg += 1
            
    print(true_pos)
    print(true_neg)
    print(false_pos)
    print(false_neg)



if __name__ == '__main__':
    test()