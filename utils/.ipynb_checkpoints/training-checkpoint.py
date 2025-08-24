import torch
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_validate(model, trainloader, validloader, criterion, optimizer, n_epoch, device, n_classes):

    train_accuracy = MulticlassAccuracy(num_classes=n_classes).to(device)
    valid_accuracy = MulticlassAccuracy(num_classes=n_classes).to(device)
    
    # Lists to store metrics for plotting
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(n_epoch):
        model.train()
        running_loss = 0.0
        step_count = 0

        for inputs, labels in tqdm(trainloader, desc=f"Training Epoch {epoch + 1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            y_hat = model(inputs) 
            loss = criterion(y_hat, labels) 
            train_accuracy.update(y_hat, labels)
            loss.backward() 
            optimizer.step() 
            running_loss += loss.item()
            step_count += 1

        training_loss = running_loss / step_count
        training_accuracy = train_accuracy.compute().item()
        train_losses.append(training_loss)
        train_accuracies.append(training_accuracy)

        model.eval()  
        valid_running_loss = 0.0
        valid_step_count = 0
        valid_accuracy.reset()  

        with torch.no_grad():
            for inputs, labels in tqdm(validloader, desc=f"Validation Epoch {epoch + 1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                y_hat = model(inputs)
                loss = criterion(y_hat, labels)
                valid_accuracy.update(y_hat, labels)
                valid_running_loss += loss.item()
                valid_step_count += 1

        validation_loss = valid_running_loss / valid_step_count
        validation_accuracy = valid_accuracy.compute().item()
        valid_losses.append(validation_loss)
        valid_accuracies.append(validation_accuracy)

        print(f"[Epoch: {epoch + 1}] training loss: {training_loss:.4f} training accuracy: {training_accuracy:.4f}")
        print(f"[Epoch: {epoch + 1}] validation loss: {validation_loss:.4f} validation accuracy: {validation_accuracy:.4f}")

        train_accuracy.reset()
        valid_accuracy.reset()

    # Seaborn settings for nicer plots
    sns.set_style("whitegrid")
    
    # Plotting the results
    plt.figure(figsize=(12, 5))

    # Plotting loss
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range(n_epoch), y=train_losses, label='Train Loss', color="salmon")
    sns.lineplot(x=range(n_epoch), y=valid_losses, label='Valid Loss', color="mediumseagreen")
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    sns.lineplot(x=range(n_epoch), y=train_accuracies, label='Train Accuracy', color="salmon")
    sns.lineplot(x=range(n_epoch), y=valid_accuracies, label='Valid Accuracy', color="mediumseagreen")
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')


    plt.tight_layout()
    plt.show()

