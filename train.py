
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

thyroid = True
cluster = True

if cluster:
    if thyroid:
        sys.path.insert(0, '/auto/home/users/m/d/mdausort/Software/large_image_source_tifffile')
    else:
        print('No need of the modified package for the moment in the brain project.')
        # sys.path.insert(0, '/CECI/home/t/g/tgodelai/large_image_source_tifffile')
elif not cluster:
    if thyroid:
        sys.path.insert(0, 'C:/Users/dausort/OneDrive - UCL/Bureau/large_image_source_tifffile')
    else:
        print('No need of the modified package for the moment in the brain project.')
        # sys.path.insert(0, 'C:/Users/tgodelaine/large_image_source_tifffile')

import large_image

# %% Cell 1 - Creation of Dataset
class ThyroidDataset(Dataset):

    def __init__(self, data_dir, annotation_file, magnification, transform=None):
        self.data_dir = data_dir
        self.annotation_path = os.path.join(self.data_dir, annotation_file)
        self.images_dir = self.data_dir
        self.transform = transform
        self.magnification = magnification
        self.classes = [1, 2]

        self.csv_file = pd.read_csv(self.annotation_path)

    def __len__(self):
        df = pd.read_csv(self.annotation_path)
        return len(df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        df = self.csv_file

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


class DBTADataset(Dataset):

    def __init__(self, data_dir, annotation_file, magnification, transform=None):
        self.data_dir = data_dir
        self.annotation_path = os.path.join(self.data_dir, annotation_file)
        self.images_dir = self.data_dir  # os.path.join(self.data_dir, 'images')
        self.transform = transform
        self.magnification = magnification
        self.classes = ['Oligodendroglioma, IDH-mutant and 1p/19q codeleted',
                        'Angiocentric glioma',
                        'Anaplastic ganglioglioma',
                        'Ganglioglioma',
                        'Desmoplastic infantile astrocytoma and ganglioglioma',
                        'Diffuse midline glioma, H3 K27M-mutant']

        self.csv_file = pd.read_csv(self.annotation_path)

    def __len__(self):
        df = pd.read_csv(self.annotation_path)
        return len(df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        df = self.csv_file

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


def train(patch_w, patch_h, stride_percent, magnification, model, name_run, bs, lr,
          num_epochs, momentum, freezed_bb, val_frequency, comment, wandb_b, patients=None):

    if thyroid:
        PATH_DATA = '/CECI/home/users/m/d/mdausort/Cytology/'
        data_dir = os.path.join(PATH_DATA, 'Training/')

        annotation_file = 'annotation_patches_' + str(patch_w) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification)
        name_project = 'proof_of_concept_cyto'
    else:
        PATH_DATA = '/CECI/home/users/t/g/tgodelai/'
        data_dir = os.path.join(PATH_DATA, 'tl/data/')

        annotation_file = 'annotation_patches_' + str(patch_w) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification) + '_' + ''.join(patients)
        # name_run = str(patch_w) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification) + '_' + ''.join(patients)
        name_project = 'test_training_with_patches'

    # Initialize wandb
    if wandb_b:
        wandb.init(project=name_project, name=name_run)
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
        config.patients = patients

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
    if thyroid:
        train_dataset = ThyroidDataset(data_dir, annotation_file + '_train.csv', magnification, transform)
        val_dataset = ThyroidDataset(data_dir, annotation_file + '_val.csv', magnification, transform)
    else:
        train_dataset = DBTADataset(data_dir, annotation_file + '_train.csv', magnification, transform)
        val_dataset = DBTADataset(data_dir, annotation_file + '_val.csv', magnification, transform)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=8)

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
    print(device)
    model_used = model_used.to(device)

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_used.parameters(), lr=lr, momentum=momentum)

    best_epoch_acc = 0.
    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Train phase
        model_used.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model_used(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Log metrics to wandb
        if wandb_b:
            wandb.log({"train_loss": epoch_loss, "train_accuracy": epoch_acc})

        print(f"Train loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        if (epoch + 1) % val_frequency == 0:

            model_used.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model_used(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(val_loader.dataset)
            epoch_acc = running_corrects.double() / len(val_loader.dataset)

            if best_epoch_acc < epoch_acc:
                best_epoch_acc = epoch_acc
                torch.save(model_used.state_dict(), os.join(PATH_DATA, name_run + '_best_epoch_weights.pth'))

            # Log metrics to wandb
            if wandb_b:
                wandb.log({"val_loss": epoch_loss, "val_accuracy": epoch_acc})

            print(f"Val loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print("Training complete")

    if wandb_b:
        wandb.save(os.join(PATH_DATA, name_run + '_best_epoch_weights.pth'))
        wandb.finish()


def test(patch_w, patch_h, stride_percent, magnification, model, name_run,
         bs, lr, comment, patients=None):

    if thyroid:
        PATH_DATA = '/CECI/home/users/m/d/mdausort/Cytology/'
        data_dir = os.path.join(PATH_DATA, 'Training/')

        annotation_file = 'annotation_patches_' + str(patch_w) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification)
        name_project = 'proof_of_concept_cyto'
    else:
        PATH_DATA = '/CECI/home/users/t/g/tgodelai/'
        data_dir = os.path.join(PATH_DATA, 'tl/data/')

        annotation_file = 'annotation_patches_' + str(patch_w) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification) + '_' + ''.join(patients)
        # name_run = str(patch_w) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification) + '_' + ''.join(patients)
        name_project = 'test_training_with_patches'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create datasets and dataloaders
    if thyroid:
        test_dataset = ThyroidDataset(data_dir, annotation_file + '_test.csv', magnification, transform)
    else:
        test_dataset = DBTADataset(data_dir, annotation_file + '_test.csv', magnification, transform)

    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

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
    model_used.load_state_dict(torch.load(os.join(PATH_DATA, name_run, 'best_epoch_weights.pth')))

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

        with torch.no_grad():
            outputs = model_used(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        test_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_inputs.extend(inputs.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_inputs = np.array(all_inputs)

    # Calculate additional metrics
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    # roc_auc = roc_auc_score(all_labels, all_preds)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    # print(f'ROC AUC: {roc_auc:.4f}')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # Add title and axis labels
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Add annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    # Save as PNG
    plt.tight_layout()
    plt.savefig(PATH_DATA + 'Testing/confusion_matrix.svg')

    # Identify worst predictions (incorrect predictions with the highest confidence)
    prediction_errors = []
    for i in range(len(all_labels)):
        if all_preds[i] != all_labels[i]:
            confidence = torch.softmax(torch.tensor(outputs[i]), dim=0)[all_preds[i]].item()
            prediction_errors.append((confidence, i))

    # Sort the errors by confidence in descending order and select the top 10
    prediction_errors.sort(reverse=True, key=lambda x: x[0])
    worst_predictions = prediction_errors[:10]

    # Plot the worst predictions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for idx, (confidence, i) in enumerate(worst_predictions):
        image = all_inputs[i].transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        true_label = all_labels[i]
        predicted_label = all_preds[i]

        axes[idx].imshow(image)
        axes[idx].set_title(f'True: {true_label}, Pred: {predicted_label}\nConf: {confidence:.2f}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(PATH_DATA + 'Testing/ten_worst_predictions.svg')


# %% Parser and launch run
def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser for the specified arguments')

    # Properties of the annotation file
    parser.add_argument('-pw', '--patch_w', type=int, default=416, help='Width of the patch')
    parser.add_argument('-ph', '--patch_h', type=int, default=416, help='Height of the patch')
    parser.add_argument('-sp', '--stride_percent', type=float, default=1.0, help='Stride percentage')
    parser.add_argument('-m', '--magnification', type=float, default=20, help='Magnification level')
    parser.add_argument("-p", "--patients", type=str, default=None, choices=[None, 'adult', 'child'], help="Patient identifier")

    # Properties of the model
    parser.add_argument('--model', type=str, choices=['resnet50', 'vgg16'], help='Name of the model')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--freezed_bb', type=int, default=1, help='Finetuning of the whole model or just the classification layer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument("--val_frequency", type=int, default=10, help="Number of epochs between validations")

    parser.add_argument('--task', type=str, choices=['train', 'test'], help='Task ask to the model')
    parser.add_argument('-com', '--comment', type=str, default='no comment', help='Specific comment on the run')
    parser.add_argument('--name_run', type=str, default='no name', help='Name of the run')
    parser.add_argument('--wandb_b', action='store_true', help='Use of wandb')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    if args.task == 'train':
        train(args.patch_w, args.patch_h, args.stride_percent, args.magnification, args.model, args.name_run,
              args.bs, args.lr, args.num_epochs, args.momentum, args.freezed_bb, args.val_frequency, args.comment, args.wandb_b, args.patients)
    elif args.task == 'test':
        test(args.patch_w, args.patch_h, args.stride_percent, args.magnification, args.model, args.name_run,
             args.bs, args.lr, args.comment, args.patients)
