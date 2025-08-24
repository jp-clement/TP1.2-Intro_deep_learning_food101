import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns

from tqdm import tqdm
from torch.utils.data import Subset
from torchvision.datasets import VisionDataset
from sklearn.metrics import confusion_matrix

def imshow(img, mean, std):
    # Reverse normalize
    img = img * torch.tensor(std).unsqueeze(1).unsqueeze(1) + torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    
    # Convert from Tensor image
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.axis('off')  # Turn off axis numbers
    

def imshow(inp, mean, std):
    """De-normalize and display an image."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

def visualize_predictions(model, dataloader, class_names, device, num_images=6):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    model.eval()  # Set model to evaluation mode
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            top3_vals, top3_preds = torch.topk(outputs, 3, dim=1)  # Get top 3 predictions

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//3, 3, images_so_far)
                ax.axis('off')
                # Include the top 3 predictions in the title
                title_str = f"True: {class_names[labels[j]]}\n" + \
                            f"1st: {class_names[top3_preds[j][0]]} ({top3_vals[j][0]:.2f})\n" + \
                            f"2nd: {class_names[top3_preds[j][1]]} ({top3_vals[j][1]:.2f})\n" + \
                            f"3rd: {class_names[top3_preds[j][2]]} ({top3_vals[j][2]:.2f})"
                ax.set_title(title_str)
                imshow(inputs.cpu().data[j], mean, std)

                if images_so_far == num_images:
                    return



def dataset_subsampling(dataset: VisionDataset, n_classes: int, n_samples_per_class: int = 100) -> Subset:
    labels = np.array(dataset._labels)

    subsets_per_class = []
    # sample n_samples_per_class
    for class_idx in range(n_classes):
        class_indices = np.where(labels==class_idx)[0][:n_samples_per_class]
        subsets_per_class.extend(class_indices)
        
    # Create a Subset from the original dataset
    subset = Subset(dataset, subsets_per_class)
    
    return subset


def compute_accuracy(model, dataloader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Computing Accuracy"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# 1. Get all predictions and targets
def get_all_preds_and_targets(model, loader, device):
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            predictions = model(inputs)
            all_preds.extend(predictions.argmax(dim=1).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_targets)

# 2. Plot the most confused pairs
def plot_most_confused(matrix, classes, num_pairs=20):
    # Zero out the diagonal
    np.fill_diagonal(matrix, 0)
    
    # Find the indices of the top N confused pairs
    sorted_indices = np.dstack(np.unravel_index(np.argsort(matrix.ravel()), matrix.shape))[0][-num_pairs:]
    
    # Extract unique class indices from the top pairs (some might be repeated)
    unique_indices = np.unique(sorted_indices)
    
    # Create a reduced matrix based on these indices
    reduced_matrix = matrix[np.ix_(unique_indices, unique_indices)]
    
    # Plot this reduced matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(reduced_matrix, annot=True, cmap="YlGnBu", 
                xticklabels=np.array(classes)[unique_indices], 
                yticklabels=np.array(classes)[unique_indices],
                fmt='g')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

