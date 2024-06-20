import large_image
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
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

cluster = True

if cluster:
    sys.path.insert(0, '/auto/home/users/m/d/mdausort/Software/large_image_source_tifffile')
else:
    sys.path.insert(0, 'C:/Users/dausort/OneDrive - UCL/Bureau/large_image_source_tifffile')

class DBTADataset(Dataset):

    def __init__(self, data_dir, annotation_file, magnification, transform=None):
        self.data_dir = data_dir
        self.annotation_path = os.path.join(self.data_dir, annotation_file)
        self.images_dir = self.data_dir
        self.transform = transform
        self.magnification = magnification
        self.classes = [1, 2]

    def __len__(self):
        images = [f for f in os.listdir(self.images_dir) if f.endswith('.ndpi')]
        return len(images)

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


def train(patch_w, patch_h, stride_percent, magnification, model, name_run,
          bs, lr, num_epochs, momentum, freezed_bb, comment, wandb_b):

    annotation_file = 'annotation_patches_' + str(patch_w) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification)

    # name_run = str(patch_w) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification)

    path_data = '/CECI/home/users/m/d/mdausort/Cytology/'
    data_dir = os.path.join(path_data, 'Training/')

    # Initialize wandb
    if wandb_b:
        wandb.init(project='proof_of_concept_cyto', name=name_run)
        config = wandb.config
        config.patch_size = (patch_w, patch_h)
        config.stride = stride_percent
        config.magnification = magnification
        config.comment = comment
        config.learning_rate = lr
        config.epochs = num_epochs
        config.momentum = momentum
        config.batch_size = bs
        config.model = model

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomChoice([
            transforms.RandomRotation(degrees=(0, 0)),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.RandomRotation(degrees=(180, 180)),
            transforms.RandomRotation(degrees=(270, 270))
        ])
    ])

    # Create datasets and dataloaders
    train_dataset = DBTADataset(data_dir, annotation_file + '_train.csv', magnification, transform)

    # Voir si ce n'est pas mieux d'enlever la magnification du dataloader et loder l'info lors de la création du csv
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_dataset = DBTADataset(data_dir, annotation_file + '_val.csv', magnification, transform)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    num_classes = len(train_dataset.classes)

    if freezed_bb == 1:
        fz_bb = False  # Freezed backbone and only the classifier layer
    else:
        fz_bb = True  # Changing backbone and only the classifier layer

    # Load the pre-trained VGG16 model
    if model == 'vgg16':
        model_used = models.vgg16(weights='IMAGENET1K_V1')
        model_used.classifier[6] = nn.Linear(model_used.classifier[6].in_features, num_classes)

        # Freeze all the layer except linear layer to finetune only the classifier layer
        for param in model_used.parameters():
            param.requires_grad = fz_bb
        for param in model_used.classifier[6].parameters():
            param.requires_grad = True

    elif model == 'resnet50':
        model_used = models.resnet50(weights='IMAGENET1K_V1')
        model_used.fc = nn.Linear(model_used.fc.in_features, num_classes)
        # Freeze all the layer except linear layer to finetune only the classifier layer
        for param in model_used.parameters():
            param.requires_grad = fz_bb
        for param in model_used.fc.parameters():
            param.requires_grad = True

    else:
        print('Model name is incorrect.')

    visualize = False
    if visualize:
        for name, layer in model_used.named_modules():
            print(name, layer)
            print(' ')

    # Move the model to the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_used = model_used.to(device)

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_used.parameters(), lr=lr, momentum=momentum)

    best_epoch_acc = 0.
    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model_used.train()
                dataloader = train_loader
            else:
                model_used.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_used(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            if best_epoch_acc < epoch_acc:
                best_epoch_acc = epoch_acc
                torch.save(model_used.state_dict(), 'best_epoch_weights.pth')

            # Log metrics to wandb
            if wandb_b:
                if phase == 'train':
                    wandb.log({'train_loss': epoch_loss, 'train_accuracy': epoch_acc})
                if phase == 'val':
                    wandb.log({'val_loss': epoch_loss, 'val_accuracy': epoch_acc})

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print('Training complete')

    if wandb_b:
        wandb.save('best_epoch_weights.pth')
        wandb.finish()


# %% Parser and launch run
def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser for the specified arguments')

    parser.add_argument('-pw', '--patch_w', type=int, default=416, help='Width of the patch')
    parser.add_argument('-ph', '--patch_h', type=int, default=416, help='Height of the patch')
    parser.add_argument('-sp', '--stride_percent', type=float, default=1.0, help='Stride percentage')
    parser.add_argument('-m', '--magnification', type=float, default=20, help='Magnification level')
    parser.add_argument('--model', type=str, default='resnet50', help='Name of the model')
    parser.add_argument('-com', '--comment', type=str, default='no comment', help='Specific comment on the run')
    parser.add_argument('--name_run', type=str, default='no name', help='Name of the run')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--freezed_bb', type=int, default=1, help='Finetuning of the whole model or just the classification layer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    # parser.add_argument('--cluster', action='store_true', help='Use of clusters')
    parser.add_argument('--wandb_b', action='store_true', help='Use of wandb')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    train(args.patch_w, args.patch_h, args.stride_percent, args.magnification, args.model, args.name_run, args.bs, args.lr,
          args.num_epochs, args.momentum, args.freezed_bb, args.comment, args.wandb_b)
