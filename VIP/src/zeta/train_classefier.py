import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import gc
from tqdm import tqdm
from datetime import datetime
import os
import json 
import logging
from glob import glob

# Assuming MultiModalityModel is defined as provided

def train_classefier_process(multi_modality_model, device, train_loader, val_loader, test_loader, config):
    # Extracting configuration parameters
    num_epochs = config['epochs']
    learning_rate = config['learning_rate']
    checkpoint_dir = config.get('checkpoint_dir', '/path/to/checkpoints')
    modalities = '_'.join(config['modalities'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_filename = f"checkpoint_{modalities}_{timestamp}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    stats_path = os.path.join(checkpoint_dir, f"stats_{modalities}_{timestamp}.json")

    # Initialize optimizer and criterion
    optimizer = optim.Adam(multi_modality_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    # Dictionary to hold training stats for each modality
    training_stats = {"epochs": [], "train_loss": {}, "val_loss": {}, "test_loss": {}}

    # Training loop
    for epoch in range(num_epochs):
        # Train and validate for each epoch
        train_losses, train_accuracies = train_epoch(multi_modality_model, device, train_loader, criterion, optimizer, epoch, num_epochs)
        val_losses, val_accuracies = evaluate_model(multi_modality_model, device, val_loader, criterion, epoch, num_epochs)

        # Update training stats
        training_stats["epochs"].append(epoch + 1)
        training_stats["train_loss"][epoch + 1] = train_losses
        training_stats["val_loss"][epoch + 1] = val_losses

        # Checkpoint logic based on overall validation loss (modify as needed for modality-specific checkpoints)
        overall_val_loss = sum(val_losses.values()) / len(val_losses)  # Average validation loss across modalities
        if overall_val_loss < best_val_loss:
            best_val_loss = overall_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': multi_modality_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'training_stats': training_stats
            }
            torch.save(checkpoint, os.path.join(checkpoint_path, 'classefier_checkpoints'))
            logging.info(f"New best model saved at epoch {epoch+1} with val loss: {best_val_loss:.4f}")

    # Save final training statistics
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f)
    logging.info(f"Training statistics saved to {stats_path}")

    # Load the best model for testing
    best_checkpoint_path = find_best_checkpoint(checkpoint_dir, modalities)
    if best_checkpoint_path:
        logging.info(f"Loading best model for testing: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path)
        multi_modality_model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate on the test set
        test_losses, test_accuracies = evaluate_model(multi_modality_model, test_loader, criterion, epoch, num_epochs)
        training_stats["test_loss"][epoch + 1] = test_losses
        logging.info(f"Test Loss: {test_losses}, Test Accuracy: {test_accuracies}")
    else:
        logging.error("No best model checkpoint found for testing.")

    print("Training, validation, and testing complete!")


def find_best_checkpoint(checkpoint_dir, modalities):
    list_of_files = glob(os.path.join(checkpoint_dir, f'checkpoint_{modalities}_*.pth'))
    if list_of_files:
        return max(list_of_files, key=os.path.getctime)
    return None


def train_epoch(model, device, train_loader, criterion, optimizer, epoch, num_epochs):
    epoch_losses = {modality: 0.0 for modality in model.module.modalities_encoders.keys()}
    epoch_accuracies = {modality: 0.0 for modality in model.module.modalities_encoders.keys()}

    model.train()
    for batch_data, batch_labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        for modality in batch_data:
            if modality in model.module.modalities_encoders:
                data = batch_data[modality].cuda(device)
                labels = batch_labels.cuda(device)

                optimizer.zero_grad()
                outputs = model.module.forward_classifier(modality, data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                accuracy = compute_accuracy(outputs, labels)
                epoch_losses[modality] += loss.item()
                epoch_accuracies[modality] += accuracy

                clear_memory()

    # Calculate average loss and accuracy for each modality
    avg_losses = {modality: epoch_losses[modality] / len(train_loader) for modality in epoch_losses}
    avg_accuracies = {modality: epoch_accuracies[modality] / len(train_loader) for modality in epoch_accuracies}

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    for modality in model.module.modalities_encoders:
        print(f"Modality: {modality}, Loss: {avg_losses[modality]:.4f}, Accuracy: {avg_accuracies[modality]:.4f}")

    return avg_losses, avg_accuracies

def evaluate_model(model, device, loader, criterion, epoch, num_epochs):
    val_losses = {modality: 0.0 for modality in model.module.modalities_encoders.keys()}
    val_accuracies = {modality: 0.0 for modality in model.module.modalities_encoders.keys()}

    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels, _ in tqdm(loader, desc=f"Validation/Test Epoch {epoch+1}/{num_epochs}"):
            for modality in batch_data:
                if modality in model.module.modalities_encoders:
                    data = batch_data[modality].cuda(device)
                    labels = batch_labels.cuda(device)

                    outputs = model.module.forward_classifier(modality, data)
                    loss = criterion(outputs, labels)

                    val_losses[modality] += loss.item()
                    val_accuracies[modality] += compute_accuracy(outputs, labels)

                    clear_memory()

    # Calculate average loss and accuracy for each modality
    avg_val_losses = {modality: val_losses[modality] / len(loader) for modality in val_losses}
    avg_val_accuracies = {modality: val_accuracies[modality] / len(loader) for modality in val_accuracies}

    print(f"Validation/Test Epoch [{epoch+1}/{num_epochs}]")
    for modality in model.modalities_encoders:
        print(f"Modality: {modality}, Loss: {avg_val_losses[modality]:.4f}, Accuracy: {avg_val_accuracies[modality]:.4f}")

    return avg_val_losses, avg_val_accuracies


def compute_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / len(labels)

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()