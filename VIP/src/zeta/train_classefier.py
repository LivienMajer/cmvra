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
import math
from sklearn.metrics import balanced_accuracy_score
import numpy as np


def train_classefier_process(multi_modality_model, device, train_loader, val_loader, test_loader, config):
    # Extracting configuration parameters
    num_epochs = config['epochs']
    learning_rate = config['learning_rate']
    checkpoint_dir = config['cktp_dir']
    modalities = '_'.join(config['modalities'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_filename = f"checkpoint_{modalities}_{timestamp}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, 'classifier_checkpoints/', checkpoint_filename)
    stats_path = os.path.join(checkpoint_dir, f"stats_{modalities}_{timestamp}.json")
    resume_from_checkpoint = config['res_cktp']

    # Initialize optimizer and criterion
    optimizer = optim.Adam(multi_modality_model.parameters(), lr=learning_rate)

    # to do implement learning rate sceduler
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize Step LR learning rate scheduler
    step_size = int(math.floor(num_epochs * 0.4))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    best_val_loss = float('inf')
    epoch = 5
    
    def find_latest_checkpoint():
        list_of_files = glob(os.path.join(checkpoint_dir,'classifier_checkpoints', f'checkpoint_{modalities}_*.pth'))
        if list_of_files:
            return max(list_of_files, key=os.path.getctime)
        return None

    # Dictionary to hold training stats for each modality
    training_stats = {"epochs": [], "train_loss": {}, "val_loss": {}, "test_loss": {}}
    start_epoch = 0
    if resume_from_checkpoint:
        latest_checkpoint_path = find_latest_checkpoint()
        if latest_checkpoint_path:
            logging.info(f"Resuming from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path)
            multi_modality_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            training_stats = checkpoint.get('training_stats', training_stats)
        else:
            logging.info("No checkpoint found, starting training from scratch.")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Train and validate for each epoch
        train_losses, train_accuracies = train_epoch(multi_modality_model, device, train_loader, criterion, optimizer, epoch, num_epochs)
        val_losses, val_accuracies = evaluate_model(multi_modality_model, device, val_loader, criterion, epoch, num_epochs)

        lr_scheduler.step()

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
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"New best model saved at epoch {epoch+1} with val loss: {best_val_loss:.4f}")

    # Save final training statistics
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f)
    logging.info(f"Training statistics saved to {stats_path}")

    # Load the best model for testing
    best_checkpoint_path = find_latest_checkpoint()
    if best_checkpoint_path:
        logging.info(f"Loading best model for testing: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path)
        multi_modality_model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate on the test set
        test_losses, test_accuracies = evaluate_model(multi_modality_model, device, test_loader, criterion, epoch, num_epochs)
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
    for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        for modality in batch_data:
            if modality in model.module.modalities_encoders:
                data = batch_data[modality].cuda(device)
                labels = batch_labels.cuda(device)
                #print(modality)
                optimizer.zero_grad()
                outputs = model.module.forward_classifier(modality, data)
                #print(f"Outputs {outputs}")
                #print(f"Labels {labels}")
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                accuracy = compute_accuracy(outputs, labels)
                #print(f"accuracy {accuracy}")
                epoch_losses[modality] += loss.item()
                epoch_accuracies[modality] += accuracy

                clear_memory()

    # Calculate average loss and accuracy for each modality
    avg_losses = {modality: epoch_losses[modality] / len(train_loader) for modality in epoch_losses}
    avg_accuracies = {modality: epoch_accuracies[modality] / len(train_loader) for modality in epoch_accuracies}

    logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
    for modality in model.module.modalities_encoders:
        logging.info(f"Modality: {modality}, Loss: {avg_losses[modality]:.4f}, Accuracy: {avg_accuracies[modality]:.4f}")

    return avg_losses, avg_accuracies

def evaluate_model(model, device, loader, criterion, epoch, num_epochs):
    val_losses = {modality: 0.0 for modality in model.module.modalities_encoders.keys()}
    val_accuracies = {modality: 0.0 for modality in model.module.modalities_encoders.keys()}

    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(loader, desc=f"Validation/Test Epoch {epoch+1}/{num_epochs}"):
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

    logging.info(f"Validation/Test Epoch [{epoch+1}/{num_epochs}]")
    for modality in model.module.modalities_encoders:
        logging.info(f"Modality: {modality}, Loss: {avg_val_losses[modality]:.4f}, Accuracy: {avg_val_accuracies[modality]:.4f}")

    return avg_val_losses, avg_val_accuracies


def compute_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / len(labels)

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def train_mae_classifier(encoder, classifier, train_data, val_data, test_data, device, cfg):
    logging.info("Starting feature extraction...")
    train_features, train_labels = extract_features(encoder, train_data, device=device, cfg=cfg)
    val_features, val_labels = extract_features(encoder, val_data, device=device, cfg=cfg)
    test_features, test_labels = extract_features(encoder, test_data, device=device, cfg=cfg)
    # Define and train the linear classifier
    input_dim = train_features.shape[1] # Adjust based on your feature size
    
    
    # Create the weighted loss function
    criterion = nn.CrossEntropyLoss()
    if cfg['dataset'] == 'DAA':
        class_counts = torch.bincount(train_labels)
        class_weights = 1. / class_counts
        class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
    optimizer = optim.Adam(classifier.parameters(), lr=cfg['learning_rate'])
    num_epochs = cfg['epochs']

    train_classifier(classifier, train_features, val_features, train_labels, val_labels, criterion, optimizer, num_epochs, batch_size=cfg['batch_size'], device=device)

    # Evaluate the classifier
    test_acc, balanced_test_acc = evaluate_classifier(classifier, test_features, test_labels, batch_size=cfg['batch_size'], device=device)
    logging.info(f"Final unblanced Test Accuracy: {test_acc}% and final balance Test Accuracy {balanced_test_acc}")

def extract_features(model, dataloader, device, cfg):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, label in tqdm(dataloader, desc="Extracting features"):
            inputs = data[cfg['modalities'][0]]# Adjust according to your data format
            inputs = inputs.to(device)

            feature = model(inputs.permute(0,2,1,3,4)) # Get features from your model
            #print(feature[0].shape)
            features.append(feature[0].cpu())
            labels.append(label)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels

def train_classifier(classifier, features, val_features, labels, val_labels, criterion, optimizer, num_epochs, batch_size, device):
    classifier.train()
    correct = 0
    total = 0
    num_samples = features.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            batch_features = features[start:end].to(device)
            batch_labels = labels[start:end].to(device)

            optimizer.zero_grad()
            outputs = classifier(batch_features)
            #print(outputs.shape)
            #print(batch_labels.shape)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')
        
        if epoch % 10 == 0:
            val_accuracy, balanced_val_acc = evaluate_classifier(classifier, val_features, val_labels, batch_size, device)
            logging.info(f'Validation Accuracy after Epoch {epoch+1}: {val_accuracy}% and balanced Validation Accuracy: {balanced_val_acc}')

def evaluate_classifier(classifier, features, labels, batch_size, device):
    classifier.eval()
    all_predictions = []
    all_labels = []
    num_samples = features.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            batch_features = features[start:end].to(device)
            batch_labels = labels[start:end].to(device)

            outputs = classifier(batch_features)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    balanced_acc = 100 * balanced_accuracy_score(all_labels, all_predictions)
    return accuracy, balanced_acc


def eval_rgb_classefier_on_ir(model, device, train_loader, val_loader, test_loader, config):
    logging.info('Evaluing the rgb classefier on ir')
    
    accuracies = {modality: 0.0 for modality in model.module.modalities_encoders.keys()}
    modality = 'ir'
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader, desc=f"Validation/Test Epoch"):
            
            if modality in model.module.modalities_encoders:
                data = batch_data[modality].cuda(device)
                labels = batch_labels.cuda(device)

                outputs = model.module.forward_encoder(modality, data)
                outputs = model.module.forward_classifier_only('rgb', outputs)                
                
                accuracies[modality] += compute_accuracy(outputs, labels)

                    

    # Calculate average loss and accuracy for each modality
    
    avg_val_accuracies = {modality: accuracies[modality] / len(test_loader) for modality in accuracies}
    logging.info(avg_val_accuracies)