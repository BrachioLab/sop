import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm.auto import tqdm
import os
import wandb



# Dataset class
class CustomDataset(Dataset):
    def __init__(self, masks, labels, transform=None):
        self.masks = masks
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mask = self.masks[idx].float()
        label = self.labels[idx]
        if self.transform:
            mask = self.transform(mask)
        return mask, label

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 512),  # Assuming input size is 224x224
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class LinearImageNetPredictor(nn.Module):
    def __init__(self, input_size=224, num_classes=1000):
        """
        A simple linear model to predict ImageNet classes from input tensors.

        Args:
            input_size (int): The height and width of the input image (assuming square images).
            num_classes (int): The number of output classes (e.g., 1000 for ImageNet).
        """
        super(LinearImageNetPredictor, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size * input_size, num_classes)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 224, 224).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Remove channel dimension by flattening after reshaping the input
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
class PatchLinearImageNetPredictor(nn.Module):
    def __init__(self, patch_size=16, num_patches=14, num_classes=1000):
        """
        A model to predict ImageNet classes from input tensors by converting masks into patch-level representations.

        Args:
            patch_size (int): The size of each patch (e.g., 16 for 16x16 patches).
            num_patches (int): The number of patches along one dimension (e.g., 14 for 14x14 patches).
            num_classes (int): The number of output classes (e.g., 1000 for ImageNet).
        """
        super(PatchLinearImageNetPredictor, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_classes = num_classes

        # Linear layer to predict based on flattened 14x14 patch representation
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_patches * num_patches, num_classes)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 224, 224).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Extract batch size
        batch_size = x.size(0)

        # Reshape input into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Shape: (batch_size, 1, num_patches, num_patches, patch_size, patch_size)

        # Compute mean value for each patch
        patches = patches.mean(dim=(-1, -2))  # Shape: (batch_size, 1, num_patches, num_patches)

        # Remove channel dimension and flatten
        patches = patches.squeeze(1).flatten(1)  # Shape: (batch_size, num_patches * num_patches)

        # Forward pass through the linear layer
        x = self.fc(patches)
        return x
    
class WrappedViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_cls = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.vit_cls(x).logits

# Training function
def train_model(model, train_loader, criterion, optimizer, device, debug=False, track=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if debug:
            if total >= 100:
                break

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)
    if track:
        wandb.log({"train_loss": avg_loss, "train_accuracy": accuracy})
    return avg_loss, accuracy

# Evaluation function
def evaluate_model(model, val_loader, criterion, device, track=False):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = val_loss / len(val_loader)
    if track:
        wandb.log({"val_loss": avg_loss, "val_accuracy": accuracy})
    return avg_loss, accuracy

def run(bin_masks_pt, labels_all_pt, model_type='cnn', lr=0.001, num_epochs=20, debug=False, seed=0, track=False):
    if track:
        # wandb.init(project='sop', entity='weiqiuy', name=f'train_group_models_{method}')
        wandb.init(
        project="sop",
        name=f"groups-{method}-{model_type}",
        config={
            "method": method,
            "learning_rate": lr,
            "num_epochs": num_epochs,
            "debug": debug,
            "model_types": model_types,
        }
    )
    # Hyperparameters
    batch_size = 64
    # num_epochs = 20
    learning_rate = lr #0.001
    num_classes = 1000

    # Prepare the dataset
    bin_masks_pt = bin_masks_pt.unsqueeze(1)  # Adding channel dimension (1 for grayscale images)
    dataset = CustomDataset(bin_masks_pt, labels_all_pt, transform=transforms.Normalize(mean=[0.5], std=[0.5]))

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset, val_dataset = random_split(dataset=dataset,
                                    lengths=[train_size, val_size],
                                    generator=generator)


    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss function, optimizer
    if model_type == 'cnn':
        model = SimpleCNN(num_classes)
    elif model_type == 'vit':
        model = WrappedViT()
    elif model_type == 'linear':
        model = LinearImageNetPredictor()
    elif model_type == 'linear_patch':
        model = PatchLinearImageNetPredictor()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        pbar.set_description(f'Epoch {epoch}')
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device, debug, track=track)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device, track=track)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    print(f"Training complete for {model_type}.")
    wandb.finish()  # Finish the wandb run for this model

    return model


def main():

    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description='Train group models')
        parser.add_argument('--method', type=str, default='agi', help='Method to train')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
        parser.add_argument('--debug', action='store_true', help='Debug mode')
        parser.add_argument('--track', action='store_true', help='track')
        
        return parser.parse_args()

    args = parse_args()
    method = args.method
    lr = args.lr
    num_epochs = args.num_epochs
    debug = args.debug
    track = args.track

    


    methods = [
        'bcos',
        'xdnn',
        'bagnet',
        'sop',
        'shap',
        'rise',
        'lime',
        'fullgrad',
        'gradcam',
        'intgrad',
        'attn',
        'archipelago',
        'mfaba',
        'agi',
        'ampe',
    ]

    if method not in methods:
        raise ValueError(f'Invalid method: {method}')

    print('Training group models for', method)

    if debug:
        model_types = ['linear_patch', 'cnn']
    else:
        model_types = ['linear_patch', 'cnn', 'vit']

    # method = 'agi'
    results_dir = f'/shared_data0/weiqiuy/sop/results/groups/imagenet_expln_d1000_m990/{method}'
    # results_dir = f'/scratch/weiqiuy/sop/results/groups/imagenet_expln_d1000_m990/{method}'

    bin_masks_pt = []
    labels_all_pt = []
    for filename in tqdm(os.listdir(results_dir)):
        if not filename.endswith('.pt'):
            continue
        results = torch.load(os.path.join(results_dir, filename))
        bin_masks_pt_curr = results['max_mask_all']
        labels_all_pt_curr = results['labels_all']
        bin_masks_pt.append(bin_masks_pt_curr)
        labels_all_pt.append(labels_all_pt_curr)

    bin_masks_pt = torch.cat(bin_masks_pt, dim=0)
    labels_all_pt = torch.cat(labels_all_pt, dim=0)
    # import pdb; pdb.set_trace()

    models = {}
    for model_type in model_types:
        print('model_type', model_type)
        model = run(bin_masks_pt, labels_all_pt, model_type=model_type, lr=lr, num_epochs=num_epochs, debug=debug, track=track)
        models[model_type] = model

    if num_epochs == 20:
        group_models_dir = '/shared_data0/weiqiuy/sop/results/groups/models'
    else:
        group_models_dir = f'/shared_data0/weiqiuy/sop/results/groups/models_{num_epochs}'
    os.makedirs(group_models_dir, exist_ok=True)
    group_models_dir_method = f'{group_models_dir}/{method}'
    os.makedirs(group_models_dir_method, exist_ok=True)
    for model_type in models:
        torch.save(models[model_type].state_dict(), 
                os.path.join(group_models_dir_method, f'{model_type}.pt'))
        print(f'Saved model {model_type} to {group_models_dir_method}')
    print('All models saved for', method)

if __name__ == '__main__':
    main()