import os
import sys
import wandb
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
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


class FeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return feature, label


def save_features_to_csv(model, data_loader, device, output_file):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, label in tqdm(data_loader, unit='batches', total=len(data_loader)):
            inputs = inputs.to(device)
            label = label.to(device)
            feature = model(inputs).cpu().numpy()
            features.append(feature)
            labels.append(label.cpu().numpy())

    # print(features.shape)
    features = np.concatenate(features)
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], features.shape[1])

    labels = np.concatenate(labels)
    if len(labels.shape) > 2:
        labels = labels.reshape(labels.shape[0], labels.shape[1])

    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(output_file, index=False)


def load_features_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    labels = data.pop('label').values
    features = data.values

    return features, labels


def train(patch_w, patch_h, condition, stride_percent, magnification, model, name_run, bs, lr,
          num_epochs, momentum, freezed_bb, val_frequency, comment, wandb_b, patients=None):

    if thyroid:
        path_data = '/CECI/home/users/m/d/mdausort/Cytology/'
        data_dir = os.path.join(path_data, 'Dataset/')

        annotation_file = 'annotation_patches_' + str(condition) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification)
        name_project = 'proof_of_concept_cyto'
    else:
        path_data = '/CECI/home/users/t/g/tgodelai/'
        data_dir = os.path.join(path_data, 'tl/data/')

        annotation_file = 'annotation_patches_' + str(condition) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification) + '_' + ''.join(patients)
        name_project = 'test_training_with_patches'

    # Initialize wandb
    if wandb_b:
        wandb.init(project=name_project, name=name_run)
        config = wandb.config
        config.patch_size = (patch_w, patch_h)
        config.condition = condition
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
        train_dataset = ThyroidDataset(data_dir, 'Annotations/' + annotation_file + '_train.csv', magnification, transform)
        val_dataset = ThyroidDataset(data_dir, 'Annotations/' + annotation_file + '_val.csv', magnification, transform)
    else:
        train_dataset = DBTADataset(data_dir, annotation_file + '_train.csv', magnification, transform)
        val_dataset = DBTADataset(data_dir, annotation_file + '_val.csv', magnification, transform)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=8)

    num_classes = len(train_dataset.classes)

    if model == 'vgg16':
        base_model = models.vgg16(weights='IMAGENET1K_V1').features
        # model_used = models.vgg16(weights='IMAGENET1K_V1')
        # model_used.classifier[6] = nn.Linear(model_used.classifier[6].in_features, num_classes)
    elif model == 'resnet50':
        base_model = nn.Sequential(*list(models.resnet50(weights='IMAGENET1K_V1').children())[:-1])
        # model_used = models.resnet50(weights='IMAGENET1K_V1')
        # model_used.fc = nn.Linear(model_used.fc.in_features, num_classes)
    elif model == 'resnet18':
        base_model = nn.Sequential(*list(models.resnet18(weights='IMAGENET1K_V1').children())[:-1])
        # model_used = models.resnet18(weights='IMAGENET1K_V1')
        # model_used.fc = nn.Linear(model_used.fc.in_features, num_classes)
    else:
        print('Model name is incorrect.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = base_model.to(device)

    features_csv = os.path.join(data_dir + 'Features/', 'features_' + str(condition) + '_' + str(model) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification) + '.csv')
    features_csv_val = os.path.join(data_dir + 'Features/', 'features_val_' + str(condition) + '_' + str(model) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification) + '.csv')

    if not os.path.exists(features_csv):
        save_features_to_csv(base_model, train_loader, device, features_csv)
    if not os.path.exists(features_csv_val):
        save_features_to_csv(base_model, val_loader, device, features_csv_val)

    features, labels = load_features_from_csv(features_csv)
    features_train_dataset = FeaturesDataset(features, labels)
    train_loader = DataLoader(features_train_dataset, batch_size=bs, shuffle=False, num_workers=8)

    features_val, labels_val = load_features_from_csv(features_csv_val)
    features_train_dataset_val = FeaturesDataset(features_val, labels_val)
    val_loader = DataLoader(features_train_dataset_val, batch_size=bs, shuffle=False, num_workers=8)

    model_linear = nn.Sequential(nn.Linear(features.shape[1], 128), nn.ReLU(), nn.Linear(128, num_classes)).to(device)

    optimizer = optim.SGD(model_linear.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    best_epoch_acc = 0.
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model_linear.train()
        running_loss = 0.0
        running_corrects = 0

        for batch, (inputs, labels) in enumerate(tqdm(train_loader, unit='batches', total=len(train_loader))):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model_linear(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        if wandb_b:
            wandb.log({"train_loss": epoch_loss, "train_accuracy": epoch_acc})

        print(f"Train loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if (epoch + 1) % val_frequency == 0:
            model_linear.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(val_loader, unit='batches', total=len(val_loader)):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model_linear(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(val_loader.dataset)
            epoch_acc = running_corrects.double() / len(val_loader.dataset)

            if best_epoch_acc < epoch_acc:
                best_epoch_acc = epoch_acc
                torch.save(model_linear.state_dict(), os.path.join(path_data + 'Train/', name_run + '_best_epoch_weights.pth'))

            if wandb_b:
                wandb.log({"val_loss": epoch_loss, "val_accuracy": epoch_acc})

            print(f"Val loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print("Training complete")

    if wandb_b:
        wandb.save(os.path.join(path_data + 'Train/', name_run + '_best_epoch_weights.pth'))
        wandb.finish()


def test(patch_w, patch_h, condition, stride_percent, magnification, model, name_run,
         bs, lr, comment, patients=None):

    if thyroid:
        path_data = '/CECI/home/users/m/d/mdausort/Cytology/'
        data_dir = os.path.join(path_data, 'Dataset/')

        annotation_file = 'annotation_patches_' + str(condition) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification)
        name_project = 'proof_of_concept_cyto'
    else:
        path_data = '/CECI/home/users/t/g/tgodelai/'
        data_dir = os.path.join(path_data, 'tl/data/')

        annotation_file = 'annotation_patches_' + str(condition) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification) + '_' + ''.join(patients)
        name_project = 'test_training_with_patches'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if thyroid:
        test_dataset = ThyroidDataset(data_dir, annotation_file + '_test.csv', magnification, transform)
    else:
        test_dataset = DBTADataset(data_dir, annotation_file + '_test.csv', magnification, transform)

    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    num_classes = len(test_dataset.classes)

    if model == 'vgg16':
        base_model = models.vgg16(weights='IMAGENET1K_V1').features
        # model_used = models.vgg16(weights='IMAGENET1K_V1')
        # model_used.classifier[6] = nn.Linear(model_used.classifier[6].in_features, num_classes)
    elif model == 'resnet50':
        base_model = nn.Sequential(*list(models.resnet50(weights='IMAGENET1K_V1').children())[:-1])
        # model_used = models.resnet50(weights='IMAGENET1K_V1')
        # model_used.fc = nn.Linear(model_used.fc.in_features, num_classes)
    elif model == 'resnet18':
        base_model = nn.Sequential(*list(models.resnet18(weights='IMAGENET1K_V1').children())[:-1])
        # model_used = models.resnet18(weights='IMAGENET1K_V1')
        # model_used.fc = nn.Linear(model_used.fc.in_features, num_classes)
    else:
        print('Model name is incorrect.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = base_model.to(device)

    features_csv_test = os.path.join(data_dir + 'Features/', 'features_test_' + str(condition) + '_' + str(model) + '_' + str(patch_h) + '_' + str(stride_percent) + '_' + str(magnification) + '.csv')

    if not os.path.exists(features_csv_test):
        save_features_to_csv(base_model, test_loader, device, features_csv_test)

    features, labels = load_features_from_csv(features_csv_test)
    features_test_dataset = FeaturesDataset(features, labels)
    test_loader = DataLoader(features_test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    model_linear = nn.Sequential(nn.Linear(features.shape[1], 128), nn.ReLU(), nn.Linear(128, num_classes)).to(device)

    # Load the best model weights
    model_linear.load_state_dict(torch.load(os.join(path_data, name_run, 'best_epoch_weights.pth')))

    # Set model to evaluation mode
    model_linear.eval()

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
            outputs = model_linear(inputs)
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
    plt.savefig(path_data + 'Test/confusion_matrix.svg')

    # Identify worst predictions (incorrect predictions with the highest confidence)
    prediction_errors = []
    for i in range(len(all_labels)):
        if all_preds[i] != all_labels[i]:
            confidence = torch.softmax(torch.tensor(outputs[i]), dim=0)[all_preds[i]].item()
            prediction_errors.append((confidence, i))

    # Sort the errors by confidence in descending order and select the top 10
    prediction_errors.sort(reverse=True, key=lambda x: x[0])
    worst_predictions = prediction_errors[: 10]

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
    plt.savefig(path_data + 'Test/ten_worst_predictions.svg')


# %% Parser and launch run
def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser for the specified arguments')

    # Properties of the annotation file
    parser.add_argument('-pw', '--patch_w', type=int, default=416, help='Width of the patch')
    parser.add_argument('-ph', '--patch_h', type=int, default=416, help='Height of the patch')
    parser.add_argument('--condition', type=str, choices=['mean', 'variance'], help='Condition to choose patches')
    parser.add_argument('-sp', '--stride_percent', type=float, default=1.0, help='Stride percentage')
    parser.add_argument('-m', '--magnification', type=float, default=20, help='Magnification level')
    parser.add_argument("-p", "--patients", type=str, default=None, choices=[None, 'adult', 'child'], help="Patient identifier")

    # Properties of the model
    parser.add_argument('--model', type=str, choices=['resnet50', 'resnet18', 'vgg16'], help='Name of the model')
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
        train(args.patch_w, args.patch_h, args.condition, args.stride_percent, args.magnification, args.model, args.name_run,
              args.bs, args.lr, args.num_epochs, args.momentum, args.freezed_bb, args.val_frequency, args.comment, args.wandb_b, args.patients)
    elif args.task == 'test':
        test(args.patch_w, args.patch_h, args.stride_percent, args.magnification, args.model, args.name_run,
             args.bs, args.lr, args.comment, args.patients)
