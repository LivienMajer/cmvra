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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np
import shutil
import random
from itertools import combinations
import torch.nn.functional as F
import matplotlib.pyplot as plt



##############################################
class MultiModalityClassifierTrainer:
    """
    A trainer class for multi-modality classifiers and their evaluation.

    This class handles the training, validation, and testing of multi-modal classifiers.
    It supports various modalities, different encoder models, and fusion techniques.

    Attributes:
        model (nn.Module): The multi-modality model to be trained.
        device (torch.device): The device to run the computations on.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        cfg (dict): Configuration dictionary containing various settings.
    """
    def __init__(self, multi_modality_model, device, train_loader, val_loader, test_loader, config):
        self.model = multi_modality_model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = config
        self.start_epoch = 0
        self.batches_per_file = 10
        self.modalities = '_'.join(self.cfg['modalities'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['learning_rate'])
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize the loss function based on the configuration
        if self.cfg.get('balancedCE', False):
            logging.info("Applying balanced Cross-Entropy loss")
            class_weights = self.compute_class_weights()
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(math.floor(self.cfg['epochs'] * 0.4)), gamma=0.5)
        self.best_val_loss = float('inf')
        stats_keys = ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'train_balanced_accuracy', 'val_balanced_accuracy']
        self.training_stats = {key: {modality: [] for modality in config['modalities']} for key in stats_keys}
        if self.cfg.get('fusion'):
            for key in stats_keys:
                self.training_stats[key]['fusion'] = []
        self.training_stats['epochs'] = []
        self.save_dir = self.cfg.get('feature_save_dir')
        if not self.save_dir:  # This will be True if self.save_dir is None or an empty string
            self.save_dir = f"/home/bas06400/Thesis/VIP/src/features/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.initialize_training()

    def compute_class_weights(self):
        """
        Compute class weights for balanced Cross-Entropy loss.

        Returns:
            torch.Tensor: Tensor of class weights.
        """
        logging.info("Computing class weights for balanced loss")
        accumulated_labels = []
        for _, label in tqdm(self.train_loader, desc="Extracting labels"):
            accumulated_labels.append(label)
        
        all_labels = torch.cat(accumulated_labels)
        class_counts = torch.bincount(all_labels)
        class_weights = 1. / class_counts.float()
        class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
        return class_weights.to(self.device)

    def initialize_training(self):
        # Setup checkpoint directory, filename, stats path, etc.
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_filename = f"checkpoint_{self.modalities}_{self.cfg['encoder_model']}_{self.cfg['dataset']}_{self.cfg['split']}_{timestamp}.pth"
        self.checkpoint_path = os.path.join(self.cfg['cktp_dir'], 'classifier_checkpoints/', checkpoint_filename)
        self.stats_path = os.path.join(self.cfg['cktp_dir'], f"stats_{self.modalities}_{timestamp}.json")
        if self.cfg['res_cktp']:
            self.resume_from_checkpoint()

    def process_epoch(self, epoch, mode='train'):
        """
        Process a single epoch for training, validation, or testing.

        This method handles the forward pass, loss computation, and backpropagation (for training)
        for each batch in the epoch.

        Args:
            epoch (int): The current epoch number.
            mode (str): One of 'train', 'val', or 'test'.

        Returns:
            dict: Metrics for the epoch, including loss and accuracy for each modality.
        """
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        modality_metrics = {modality: {'total_loss': 0, 'predictions': [], 'labels': []} for modality in self.cfg['modalities']}
        #total_batches_processed = 0

        for modality in self.cfg['modalities']:
            file_prefix1 = f"{mode}"
            file_prefix2 = f"{mode}_{modality}"
            num_files = self.cfg['num_files'][modality][mode]
            #print(f'num_files {num_files}')

            for file_index in range(num_files):
                feature_file = os.path.join(self.save_dir, file_prefix1,f"{file_prefix2}_features_{file_index}.pt")
                label_file = os.path.join(self.save_dir, file_prefix1,f"{file_prefix2}_labels_{file_index}.pt")
                features = torch.load(feature_file).to(self.device)
                labels = torch.load(label_file).to(self.device)
                #print(f'feature_size {features.shape}')
                for start in range(0, features.size(0), self.cfg['batch_size']):
                    
                    end = start + self.cfg['batch_size']
                    batch_features = features[start:end]
                    batch_labels = labels[start:end]

                    if mode == 'train':
                        self.optimizer.zero_grad()

                    outputs = self.model.module.forward_classifier_only(modality ,batch_features)
                    loss = self.criterion(outputs, batch_labels)

                    if mode == 'train':
                        loss.backward()
                        self.optimizer.step()

                    modality_metrics[modality]['total_loss'] += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    modality_metrics[modality]['predictions'].extend(predicted.cpu().numpy())
                    modality_metrics[modality]['labels'].extend(batch_labels.cpu().numpy())
                    #total_batches_processed += 1  # Increment the batch counter

        # Log the total number of batches processed for this epoch and mode
        #logging.info(f'Epoch {epoch+1}, {mode.capitalize()} - Total Batches Processed: {total_batches_processed}')


        # After processing all batches, calculate and log metrics for each modality
        for modality, metrics in modality_metrics.items():
            accuracy = accuracy_score(metrics['labels'], metrics['predictions'])
            balanced_acc = balanced_accuracy_score(metrics['labels'], metrics['predictions'])
            logging.info(f'Epoch {epoch+1}, {mode.capitalize()} {modality} - Loss: {metrics["total_loss"] / len(metrics["labels"]):.4f}, '
                         f'Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_acc:.4f}')

        epoch_metrics = {
            modality: {
                'loss': metrics['total_loss'] / len(metrics['labels']),
                'accuracy': accuracy_score(metrics['labels'], metrics['predictions']),
                'balanced_accuracy': balanced_accuracy_score(metrics['labels'], metrics['predictions'])
            } for modality, metrics in modality_metrics.items()
        }
        # Save test metrics if flag is set and mode is test
        if mode == 'test' and self.cfg.get('save_test_metrics', False):
            self.save_test_metrics(modality_metrics)
        return epoch_metrics


    def train(self):
        """
        Main training loop for the multi-modality classifier.

        This method handles the entire training process, including:
        - Extracting features if necessary
        - Training for the specified number of epochs
        - Validating the model after each epoch
        - Saving checkpoints and statistics
        - Evaluating on the test set after training
        """
        if self.cfg['res_cktp']:
            self.resume_from_checkpoint()
        else:
            if not self.cfg['full_train_classifiers']:
                self.extract_and_save_all_features()
            else:
                logging.info("Full Training enabled")
        for epoch in range(self.start_epoch, self.cfg['epochs']):
            if self.cfg['fusion']:
                train_metrics = self.process_epoch_fusion(epoch, mode='train')
                val_metrics = self.process_epoch_fusion(epoch, mode='val')
                # Here, process_epoch should return metrics in a structure that update_training_stats expects
                self.update_training_stats(epoch, train_metrics, val_metrics, fusion=True)
            else:
                if not self.cfg['full_train_classifiers']:
                    train_metrics = self.process_epoch(epoch, mode='train')
                    val_metrics = self.process_epoch(epoch, mode='val')
                else:
                    train_metrics = self.process_epoch_full_training(epoch, mode='train')
                    val_metrics = self.process_epoch_full_training(epoch, mode='val')
                # Here, process_epoch should return metrics in a structure that update_training_stats expects
                self.update_training_stats(epoch, train_metrics, val_metrics)

            overall_val_loss = sum(val_metrics[modality]['loss'] for modality in self.cfg['modalities']) / len(self.cfg['modalities']) if not self.cfg['fusion'] else val_metrics['loss']
            if overall_val_loss < self.best_val_loss:
                self.best_val_loss = overall_val_loss
                self.save_checkpoint(epoch)
        
            self.lr_scheduler.step()
        if self.cfg['epochs'] == 0:
            self.save_checkpoint(0)
        self.save_training_stats()
        self.resume_from_checkpoint()
        self.evaluate_test_set()
        self.delete_saved_features_dir()

    def process_epoch_full_training(self, epoch, mode='train'):
        """
        Process a single epoch for full training (without pre-extracted features).

        This method is used when training the entire model end-to-end, including feature extraction.

        Args:
            epoch (int): The current epoch number.
            mode (str): One of 'train', 'val', or 'test'.

        Returns:
            dict: Metrics for the epoch, including loss and accuracy for each modality.
        """
        if mode == 'train':
            self.model.train()
            dataloader = self.train_loader  
        elif mode == 'val':
            self.model.eval()
            dataloader = self.val_loader  
        elif mode == 'test':
            self.model.eval()
            dataloader = self.test_loader  
        else:
            raise ValueError("Invalid mode. Expected one of: 'train', 'val', 'test'.")
        modality_metrics = {modality: {'total_loss': 0, 'predictions': [], 'labels': []} for modality in self.cfg['modalities']}
        
        with torch.no_grad() if mode != 'train' else torch.enable_grad():
            for batch_data, batch_labels in tqdm(dataloader, desc=f"Epoch {epoch+1} - {mode.capitalize()}"):
                for modality in self.cfg['modalities']:
                    if modality in self.model.module.modalities_encoders:
                        inputs = batch_data[modality].to(self.device)
                        labels = batch_labels.to(self.device)

                        
                        if self.cfg['encoder_model'] in ['CLIP-VIP', 'MAE']:
                            if self.cfg['encoder_model'] == 'MAE':
                                inputs = inputs.permute(0, 2, 1, 3, 4)  
                            outputs = self.model.module.forward_classifier(modality, inputs)
                        elif self.cfg['encoder_model'] == 'MIX':
                            encoder_type = self.cfg['modalities_encoders'][modality]
                            inputs = self.preprocess_data(inputs, modality, encoder_type)  # Preprocess based on mixed encoder type
                            outputs = self.model.module.forward_classifier(modality, inputs)
                        else:
                            logging.info(f"Unsupported encoder model: {self.cfg['encoder_model']}")
                            continue

                        loss = self.criterion(outputs, labels)

                        if mode == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                        modality_metrics[modality]['total_loss'] += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        modality_metrics[modality]['predictions'].extend(predicted.cpu().numpy())
                        modality_metrics[modality]['labels'].extend(labels.cpu().numpy())

        # Calculate and log metrics for each modality
        for modality, metrics in modality_metrics.items():
            accuracy = accuracy_score(metrics['labels'], metrics['predictions'])
            balanced_acc = balanced_accuracy_score(metrics['labels'], metrics['predictions'])
            logging.info(f'Epoch {epoch+1}, {mode.capitalize()} {modality} - Loss: {metrics["total_loss"] / len(metrics["labels"]):.4f}, '
                        f'Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_acc:.4f}')

        epoch_metrics = {
            modality: {
                'loss': metrics['total_loss'] / len(metrics["labels"]),
                'accuracy': accuracy_score(metrics['labels'], metrics['predictions']),
                'balanced_accuracy': balanced_accuracy_score(metrics['labels'], metrics['predictions'])
            } for modality, metrics in modality_metrics.items()
        }

        return epoch_metrics
    
    def process_epoch_fusion(self, epoch, mode='train', zeroing_probability=0.1):
        """
        Process a single epoch for fusion-based training.

        This method handles the processing for models that use feature fusion across modalities.

        Args:
            epoch (int): The current epoch number.
            mode (str): One of 'train', 'val', or 'test'.
            zeroing_probability (float): Probability of zeroing out a modality during training.

        Returns:
            dict: Metrics for the epoch, including loss and accuracy for the fused model.
        """
        
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_predictions = []

        
        test_results = {}
        # old modality_combinations = ['all'] + self.cfg['modalities'] if mode == 'test' else ['all']

        if mode == 'test':
            modality_indices = range(len(self.cfg['modalities']))
            modality_combinations = ['all']  # Start with 'all' (no modalities blocked)
            # Generate combinations, excluding the one that includes all modalities
            for r in range(1, len(self.cfg['modalities'])):  # Up to but not including all modalities
                modality_combinations += ['+'.join(self.cfg['modalities'][idx] for idx in combo) for combo in combinations(modality_indices, r)]
        else:
            modality_combinations = ['all']
        num_files = self.cfg['num_files'][self.cfg['modalities'][-1]][mode]

        for combination in modality_combinations:
            for file_index in range(num_files):
                # Construct file paths for features and labels
                feature_files = [os.path.join(self.save_dir, mode, f"{mode}_{modality}_features_{file_index}.pt") for modality in self.cfg['modalities']]
                label_file = os.path.join(self.save_dir, mode, f"{mode}_{self.cfg['modalities'][-1]}_labels_{file_index}.pt")

                # Load features and labels
                features = [torch.load(f).to(self.device).unsqueeze(1) for f in feature_files]
                labels = torch.load(label_file).to(self.device)

                for start in range(0, labels.size(0), self.cfg['batch_size']):
                    end = start + self.cfg['batch_size']
                    batch_features = [f[start:end] for f in features]
                    batch_labels = labels[start:end]

                    if mode == 'train':
                        # Randomly zero out modalities for training
                        for i in range(len(batch_features)):
                            if random.random() < zeroing_probability:
                                batch_features[i] = torch.zeros_like(batch_features[i])
                    elif mode == 'test':
                        # Systematically zero out one modality for testing
                        #if combination != 'all':
                        #    batch_features[self.cfg['modalities'].index(combination)] = torch.zeros_like(batch_features[self.cfg['modalities'].index(combination)])
                        if combination != 'all':
                            modalities_to_block = combination.split('+')  # Assuming combination like 'mod1+mod2'
                            for modality in modalities_to_block:
                                index = self.cfg['modalities'].index(modality)
                                batch_features[index] = torch.zeros_like(batch_features[index])
                    # Concatenate features along the last dimension
                    concatenated_features = torch.cat(batch_features, dim=1)

                    # Forward pass
                    outputs = self.model.module.forward_fusion(concatenated_features)
                    loss = self.criterion(outputs, batch_labels)

                    if mode == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    total_loss += loss.item() * batch_labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == batch_labels).sum().item()
                    total_samples += batch_labels.size(0)
                    all_labels.extend(batch_labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            average_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
            # Log epoch metrics
            logging.info(f'Epoch {epoch+1}, {mode.capitalize()} - {combination} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_acc:.4f}')
            test_results[combination] = {'loss': average_loss, 'accuracy': accuracy, 'balanced_accuracy': balanced_acc}

            # Reset metrics for the next modality combination
            total_loss = 0
            total_correct = 0
            total_samples = 0
            all_labels = []
            all_predictions = []
        
        return test_results if mode == 'test' else {'loss': average_loss, 'accuracy': accuracy}

    def evaluate_test_set(self):

        if self.cfg.get('full_train_classifiers', False):
            logging.info("Using full training classifiers")
            test_metrics = self.process_epoch_full_training(0, mode='test')
        elif self.cfg.get('fusion', False):
            logging.info("Using fusion model for evaluation")
            test_metrics = self.process_epoch_fusion(0, mode='test')
        else:
            logging.info("Using standard test processing")
            test_metrics = self.process_epoch(0, mode='test')

    def update_training_stats(self, epoch, train_metrics, val_metrics, fusion=False):
        self.training_stats["epochs"].append(epoch)

        if fusion:
            self.training_stats['train_loss']['fusion'].append(train_metrics['loss'])
            self.training_stats['train_accuracy']['fusion'].append(train_metrics['accuracy'])
            self.training_stats['val_loss']['fusion'].append(val_metrics['loss'])
            self.training_stats['val_accuracy']['fusion'].append(val_metrics['accuracy'])
        else:
            for modality in self.cfg['modalities']:
                # Ensure initialization for each modality if not already done
                for key in ["train_loss", "train_accuracy", "train_balanced_accuracy", "val_loss", "val_accuracy", "val_balanced_accuracy"]:
                    if modality not in self.training_stats[key]:
                        self.training_stats[key][modality] = []
                
                # Update training stats with metrics from process_epoch
                self.training_stats["train_loss"][modality].append(train_metrics[modality]['loss'])
                self.training_stats["train_accuracy"][modality].append(train_metrics[modality]['accuracy'])
                self.training_stats["train_balanced_accuracy"][modality].append(train_metrics[modality]['balanced_accuracy'])
                
                self.training_stats["val_loss"][modality].append(val_metrics[modality]['loss'])
                self.training_stats["val_accuracy"][modality].append(val_metrics[modality]['accuracy'])
                self.training_stats["val_balanced_accuracy"][modality].append(val_metrics[modality]['balanced_accuracy'])

    def retrieve_extracted_file_counts(self):
        """
        Scans the feature save directories for each modality and dataset mode
        to update the self.cfg['num_files'] dictionary with the actual number of
        extracted feature files present.
        """
        # Initialize num_files if not present
        num_files = self.cfg.get('num_files', {})
        if not num_files:
            self.cfg['num_files'] = {modality: {'train': 0, 'val': 0, 'test': 0} for modality in self.cfg['modalities']}
        for mode in ['train', 'val', 'test']:
            for modality in self.cfg['modalities']:
                # Construct the directory path for the current mode and modality
                dir_path = os.path.join(self.save_dir, mode)
                # Pattern to match files for the current modality
                file_pattern = f"{mode}_{modality}_features_*.pt"
                # List all matching feature files
                feature_files = glob(os.path.join(dir_path, file_pattern))
                # Update the count in self.cfg['num_files']
                self.cfg['num_files'][modality][mode] = len(feature_files)

        logging.info("Updated file counts from saved features.")

    def resume_from_checkpoint(self, specific_checkpoint_path=None):
        """
        Resumes training from a specific checkpoint or the latest best checkpoint.

        Args:
            specific_checkpoint_path (str, optional): Path to a specific checkpoint file to resume from. 
                                                       If None, resumes from the latest best checkpoint.
        """
        checkpoint_path = specific_checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()

        if checkpoint_path and os.path.isfile(checkpoint_path):
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{self.device}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint.get('epoch', self.start_epoch)
            self.best_val_loss = checkpoint.get('best_val_loss', self.best_val_loss)
            # Optionally, also restore LR scheduler state
            if 'lr_scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            logging.info(f"Resumed from checkpoint: {checkpoint_path}")
            self.retrieve_extracted_file_counts()
        else:
            logging.error("No checkpoint found to resume from. Starting the Classifier training from scratch")
            self.retrieve_extracted_file_counts()

    def find_latest_checkpoint(self):
        """
        Finds the latest (best) checkpoint file in the checkpoint directory.

        Returns:
            str: Path to the latest checkpoint file, or None if no checkpoint found.
        """
        checkpoint_dir = os.path.join(self.cfg['cktp_dir'], 'classifier_checkpoints')
        list_of_files = glob(os.path.join(checkpoint_dir, f'checkpoint_{self.modalities}_{self.cfg["encoder_model"]}_{self.cfg["dataset"]}_{self.cfg["split"]}_*.pth'))
        if list_of_files:
            latest_checkpoint = max(list_of_files, key=os.path.getctime)
            return latest_checkpoint
        return None
    
    def save_training_stats(self):
        """
        Saves the training statistics to a JSON file.
        """
        with open(self.stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=4)
        logging.info(f"Training statistics saved to {self.stats_path}")

    def extract_and_save_all_features(self):
        """
        Extracts and saves features for training, validation, and test datasets.

        Args:
            batches_per_file (int): Number of batches to aggregate before saving to disk.
            save_dir (str): Base directory to save extracted features.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Initialize or reset the num_files dictionary
        self.cfg['num_files'] = {modality: {'train': 0, 'val': 0, 'test': 0} for modality in self.cfg['modalities']}

        # Extract and save features for each dataset
        for mode in ['train', 'val', 'test']:
            loader = getattr(self, f"{mode}_loader", None)
            if loader is not None:
                num_files = self.extract_features_and_save(loader, self.cfg, mode, self.batches_per_file, os.path.join(self.save_dir, mode))
                for modality in self.cfg['modalities']:
                    # Update the num_files for each modality and mode
                    self.cfg['num_files'][modality][mode] = num_files

    def preprocess_data(self, inputs, modality, encoder_type):
        # Implement preprocessing based on encoder type
        if encoder_type == 'MAE':
            return inputs.permute(0, 2, 1, 3, 4)  # Example for MAE
        elif encoder_type == 'OMNIVORE':
            if modality =='depth':
                return torch.cat((inputs, inputs[:, :, 0:1, :, :]), 2).permute(0, 2, 1, 3, 4)
            else:
                return inputs.permute(0, 2, 1, 3, 4)
        elif encoder_type == 'MAEPS':
            return inputs.view(inputs.size(0),inputs.size(1), -1)
        return inputs  # Default case (e.g., CLIP-VIP and DINO do not need special preprocessing)
    
    def extract_features_and_save(self, dataloader, cfg, file_prefix, batches_per_file, save_dir):
        """
        Extract features for all datasets (train, val, test) and save them to disk.

        This method is used to precompute features, which can speed up subsequent training.
        """
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.eval()
        batch_count = 0
        file_index = 0

        # Assert that every modality in cfg['modalities'] has an entry in cfg['modalities_encoders'] when using 'MIX'
        if cfg['encoder_model'] == 'MIX':
            assert all(modality in cfg['modalities_encoders'] for modality in cfg['modalities']), \
                "Each modality in cfg['modalities'] must have a corresponding encoder in cfg['modalities_encoders'] when using 'MIX'."

        # Containers for accumulating features and labels across batches
        accumulated_features = {modality: [] for modality in cfg['modalities'] if modality in self.model.module.modalities_encoders}
        accumulated_labels = []

        with torch.no_grad():
            for data, label in tqdm(dataloader, desc="Extracting features"):
                for modality in cfg['modalities']:
                    if modality in self.model.module.modalities_encoders:
                        inputs = data[modality].to(self.device)

                        # Adjust for different encoder models
                        if cfg['encoder_model'] == 'CLIP-VIP':
                            feature = self.model.module.forward_encoder(modality, inputs)
                        elif cfg['encoder_model'] == 'MAE':
                            feature = self.model.module.forward_encoder(modality, inputs.permute(0, 2, 1, 3, 4))
                        elif cfg['encoder_model'] == 'MIX':
                            inputs = self.preprocess_data(inputs, modality, cfg['modalities_encoders'][modality])
                            feature = self.model.module.forward_encoder(modality, inputs)
                        else:
                            logging.info(f"Unsupported encoder model: {cfg['encoder_model']}")
                            continue

                        accumulated_features[modality].append(feature.cpu())

                accumulated_labels.append(label)

                batch_count += 1
                if batch_count == batches_per_file:
                    # Concatenate and save when batches_per_file is reached
                    for modality, features_list in accumulated_features.items():
                        concatenated_features = torch.cat(features_list, dim=0)
                        concatenated_labels = torch.cat(accumulated_labels, dim=0)
                        self._save_batch(concatenated_features, concatenated_labels, save_dir, f"{file_prefix}_{modality}", file_index)
                    
                    # Reset for next group of batches
                    batch_count = 0
                    file_index += 1
                    accumulated_features = {modality: [] for modality in accumulated_features}
                    accumulated_labels = []

            # Check and save any remaining data not reaching batches_per_file
            if batch_count > 0:  # This checks if there are unsaved features
                for modality, features_list in accumulated_features.items():
                    if features_list:  # Ensure there's data to save
                        concatenated_features = torch.cat(features_list, dim=0)
                        concatenated_labels = torch.cat(accumulated_labels, dim=0)
                        self._save_batch(concatenated_features, concatenated_labels, save_dir, f"{file_prefix}_{modality}", file_index)

        return file_index + 1 if batch_count > 0 else file_index

    def _save_batch(self, features, labels, save_dir, file_prefix, file_index):
        feature_file = os.path.join(save_dir, f"{file_prefix}_features_{file_index}.pt")
        label_file = os.path.join(save_dir, f"{file_prefix}_labels_{file_index}.pt")
        torch.save(features, feature_file)
        torch.save(labels, label_file)


    def delete_saved_features_dir(self):
    # Check if the directory exists
        if os.path.exists(self.save_dir):
            # Use shutil.rmtree to delete the directory and all its contents
            shutil.rmtree(self.save_dir)
            print(f"Deleted the directory and all contents: {self.save_dir}")
        else:
            print(f"The directory does not exist: {self.save_dir}")
   
    def save_checkpoint(self, epoch):
        """
        Saves a checkpoint at the specified epoch.

        Args:
            epoch (int): The current epoch number.
        """
        checkpoint = {
            'epoch': epoch + 1,  # Saving next epoch to start from
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()  # Optional: Save LR scheduler state
        }
        # Save checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        logging.info(f"Checkpoint saved at epoch {epoch} to {self.checkpoint_path}")

    def compute_class_stats(self, embeddings, labels):
        """
        Compute the centers and covariance matrices for each class in the embeddings.
        
        Args:
            embeddings (torch.Tensor): The embeddings tensor, shape (num_samples, embed_dim)
            labels (torch.Tensor): The corresponding labels tensor, shape (num_samples,)
        
        Returns:
            dict: A dictionary containing 'centers' and 'covariances' for each class.
        """
        unique_labels = labels.unique()
        class_stats = {}
        for label in unique_labels:
            class_embeddings = embeddings[labels == label]
            center = class_embeddings.mean(dim=0)
            if class_embeddings.shape[0] > 1:
                covariance = torch.cov(class_embeddings.T)
            else:
                covariance = torch.zeros((class_embeddings.shape[1], class_embeddings.shape[1]))
            class_stats[label.item()] = {'center': center, 'covariance': covariance}
        return class_stats
    
    def cosine_distance(self, x1, x2):
        """
        Compute the cosine distance between two vectors.
        
        Args:
            x1 (torch.Tensor): Vector 1, shape (embed_dim,)
            x2 (torch.Tensor): Vector 2, shape (embed_dim,)
        
        Returns:
            float: Cosine distance between the two vectors.
        """
        cos_sim = F.cosine_similarity(x1.unsqueeze(0), x2.unsqueeze(0), dim=1)
        return 1 - cos_sim.item()

    def compare_embeddings(self, stats1, stats2):
        """
        Compare two sets of embedding statistics. This includes comparing centers using
        Euclidean distance, cosine distance, and differences in covariance matrices.
        
        Args:
            stats1, stats2 (dict): Dictionaries of class statistics as returned by `compute_class_stats`.
        
        Returns:
            dict: A dictionary of comparison results including center distances and covariance differences.
        """
        comparison_results = {}
        for label in stats1:
            if label in stats2:
                center_distance = torch.norm(stats1[label]['center'] - stats2[label]['center']).item()
                cosine_dist = self.cosine_distance(stats1[label]['center'], stats2[label]['center'])
                cov_diff = torch.norm(stats1[label]['covariance'] - stats2[label]['covariance']).item()
                comparison_results[label] = {
                    'center_distance': center_distance,
                    'cosine_distance': cosine_dist,
                    'covariance_difference': cov_diff
                }
            else:
                comparison_results[label] = {
                    'center_distance': None,
                    'cosine_distance': None,
                    'covariance_difference': None
                }
        return comparison_results
    
    def collect_embeddings(self, epoch):
        self.model.eval()  # Set model to evaluation mode

        modality_embeddings = {modality: [] for modality in self.cfg['modalities']}
        modality_labels = {modality: [] for modality in self.cfg['modalities']}

        for modality in self.cfg['modalities']:
            file_prefix = f"test_{modality}"
            num_files = self.cfg['num_files'][modality]['test']

            for file_index in range(num_files):
                feature_file = os.path.join(self.save_dir, "test", f"{file_prefix}_features_{file_index}.pt")
                label_file = os.path.join(self.save_dir, "test", f"{file_prefix}_labels_{file_index}.pt")
                features = torch.load(feature_file)
                labels = torch.load(label_file)

                
                for start in range(0, features.size(0), self.cfg['batch_size']):
                    end = start + self.cfg['batch_size']
                    embeddings = features[start:end]
                    batch_labels = labels[start:end]
                    modality_embeddings[modality].extend(embeddings.cpu().numpy())
                    modality_labels[modality].extend(batch_labels.cpu().numpy())

        # You may process or save the embeddings and labels here, depending on your downstream task
        embeddings_info = {
            modality: {
                'embeddings': np.array(modality_embeddings[modality]),
                'labels': np.array(modality_labels[modality])
            } for modality in modality_embeddings
        }

        return embeddings_info
    
    def save_test_metrics(self, metrics):
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        serializable_metrics = json.loads(
            json.dumps(metrics, default=convert_to_serializable)
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_metrics_{self.modalities}_{self.cfg['encoder_model']}_{self.cfg['dataset']}_{self.cfg['split']}_{timestamp}.json"
        filepath = os.path.join('/home/bas06400/Thesis/VIP/src/predictions', filename)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        logging.info(f"Test metrics saved to {filepath}")
    
    
    def analyze_embeddings(self, text_embeddings):
        """
        Collects embeddings, computes their statistics, and compares them across modalities and with text embeddings.

        Args:
            text_embeddings (torch.Tensor): Pre-computed text embeddings.

        Returns:
            dict: A dictionary containing comparison results and statistics for all modalities and text embeddings.
        """
        if self.cfg['res_cktp']:
            self.resume_from_checkpoint()
        else:
            self.extract_and_save_all_features()
        # Step 1: Collect embeddings for each modality
        modality_embeddings_info = self.collect_embeddings(epoch=None)  

        # Step 2: Compute statistics for each modality
        modality_stats = {}
        for modality, data in modality_embeddings_info.items():
            modality_stats[modality] = self.compute_class_stats(torch.tensor(data['embeddings']), torch.tensor(data['labels']))

        # Step 3: Compute statistics for text embeddings
        text_stats = self.compute_class_stats(torch.tensor(text_embeddings),torch.arange(0, torch.tensor(text_embeddings).size(0)))

    

        # Step 4: Collect cosine distances for individual samples
        cosine_distances = {modality: [] for modality in modality_stats}
        text_embeddings_tensor = torch.tensor(text_embeddings)
        for modality, data in modality_embeddings_info.items():
            embeddings = torch.tensor(data['embeddings'])
            for emb in embeddings:
                for text_emb in text_embeddings_tensor:
                    cos_dist = self.cosine_distance(emb, text_emb)
                    cosine_distances[modality].append(cos_dist)

        # Step 5: Create histograms with distinct colors
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed
        plt.figure(figsize=(10, 6))
        for idx, (modality, distances) in enumerate(cosine_distances.items()):
            plt.hist(distances, bins=np.arange(0.5, 1.5, 0.001), alpha=0.5, label=f'{modality} vs Text', color=colors[idx % len(colors)])

        plt.xlabel('Cosine Distance')
        plt.ylabel('Frequency')
        plt.title('Histogram of Cosine Distances')
        plt.legend(loc='upper right')
        plt.grid(True)

        histogram_path = os.path.join(self.save_dir, 'cosine_distance_histograms_all_distances.png')
        plt.savefig(histogram_path)
        plt.close()
        logging.info(f"Histogram saved at {histogram_path}")
         # Step 4: Collect cosine similarities and compute ratios for each sample
        ratios = {modality: [] for modality in modality_stats}
        text_embeddings_tensor = torch.tensor(text_embeddings)
        for modality, data in modality_embeddings_info.items():
            embeddings = torch.tensor(data['embeddings'])
            labels = torch.tensor(data['labels'])
            for emb, label in zip(embeddings, labels):
                true_class_text_emb = text_embeddings_tensor[label]
                cos_sim_true_class = self.cosine_distance(emb, true_class_text_emb)
                other_text_embeddings = torch.cat([text_embeddings_tensor[:label], text_embeddings_tensor[label+1:]])
                cos_sim_other_classes = [self.cosine_distance(emb, other_emb) for other_emb in other_text_embeddings]
                avg_cos_sim_other_classes = np.mean(cos_sim_other_classes)
                ratio = cos_sim_true_class / avg_cos_sim_other_classes
                ratios[modality].append(ratio)

        # Debug: Log ratios
        for modality, ratio_values in ratios.items():
            logging.debug(f"{modality} ratios: {ratio_values}")

        # Step 5: Create histograms for the ratios of all samples for each modality
        plt.figure(figsize=(10, 6))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  
        for idx, (modality, ratio_values) in enumerate(ratios.items()):
            plt.hist(ratio_values, bins=np.arange(0.5, 1.5, 0.001), alpha=0.5, label=f'{modality}', color=colors[idx % len(colors)])

        plt.xlabel('Ratio (True Class Cosine Distance / Average Other Classes Cosine Distance)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Cosine Similarity Ratios for All Samples')
        plt.legend(loc='upper right')
        plt.grid(True)

        histogram_path = os.path.join(self.save_dir, 'cosine_Distance_ratio_histograms_all_samples.png')
        plt.savefig(histogram_path)
        plt.close()
        logging.info(f"Histogram saved at {histogram_path}")

        # Compare statistics between each modality and text embeddings
        for modality in modality_stats:
            comparison_result = self.compare_embeddings(modality_stats[modality], text_stats)
            logging.info(f"Comparison {modality}_vs_text: {comparison_result}")

        # Compare modalities against each other
        for modality1 in modality_stats:
            for modality2 in modality_stats:
                if modality1 != modality2:
                    key = f"{modality1}_vs_{modality2}"
                    comparison_result = self.compare_embeddings(modality_stats[modality1], modality_stats[modality2])
                    logging.info(f"Comparison {key}: {comparison_result}")

        return 



#################################################################################################################
def train_mae_classifier2(encoder, classifier, train_data, val_data, test_data, device, cfg):
    logging.info("Starting feature extraction...")
    train_features, train_labels = extract_features(encoder, train_data, device=device, cfg=cfg)
    val_features, val_labels = extract_features(encoder, val_data, device=device, cfg=cfg)
    test_features, test_labels = extract_features(encoder, test_data, device=device, cfg=cfg)
    # Define and train the linear classifier
    input_dim = train_features.shape[1] 
    
    
    # Create the weighted loss function
    criterion = nn.CrossEntropyLoss()
    if cfg['dataset'] == 'DAA':
        logging.info("Applying balance loss")
        class_counts = torch.bincount(train_labels)
        class_weights = 1. / class_counts
        class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
    optimizer = optim.Adam(classifier.parameters(), lr=cfg['learning_rate'])
    num_epochs = cfg['epochs']

    train_classifier(classifier, train_features, val_features, train_labels, val_labels, criterion, optimizer, num_epochs, batch_size=cfg['batch_size'], device=device)

    # Evaluate the classifier
    test_acc, balanced_test_acc = evaluate_classifier(classifier, test_features, test_labels, batch_size=cfg['batch_size'], device=device)
    logging.info(f"Final unblanced test accuracy: {test_acc}% and final balanced test accuracy {balanced_test_acc}")

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
            logging.info(f'Validation accuracy after epoch {epoch+1}: {val_accuracy}% and balanced validation accuracy: {balanced_val_acc}')

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
    #unique_predictions = np.unique(all_predictions)
    #unique_labels = np.unique(all_labels)
    #print(f"Unique predicted classes: {unique_predictions}")
    #print(f"Unique true classes: {unique_labels}")
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







####################################################
# saving step
def train_mae_classifier(encoder, classifier, train_data, val_data, test_data, device, cfg, save_dir='/home/bas06400/Thesis/VIP/src/features/feat1'):
    logging.info("Starting feature extraction...")
    num_train_files = extract_features2(encoder, train_data, device, cfg, 'train', 10, save_dir)
    num_val_files = extract_features2(encoder, val_data, device, cfg, 'val', 10, save_dir)
    num_test_files = extract_features2(encoder, test_data, device, cfg, 'test', 10, save_dir)
    
    
    
    # Create the weighted loss function
    criterion = nn.CrossEntropyLoss()
    if cfg['dataset'] == 'DAA':
        logging.info("Applying balanced loss")
        class_counts = calculate_class_counts('train', num_train_files, save_dir)
        class_weights = 1. / class_counts.float()
        class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.Adam(classifier.parameters(), lr=cfg['learning_rate'])

    train_classifier2(classifier, 'train', num_train_files, num_val_files, criterion, optimizer, cfg['epochs'], cfg['batch_size'], device, save_dir)
    test_acc, balanced_test_acc = evaluate_classifier2(classifier, 'test', num_test_files, cfg['batch_size'], device, save_dir)

    # Cleaning up files
    delete_feature_files('train', num_train_files, save_dir)
    delete_feature_files('val', num_val_files, save_dir)
    delete_feature_files('test', num_test_files, save_dir)

    logging.info(f"Final unblanced test accuracy: {test_acc}% and final balanced test accuracy {balanced_test_acc}")

def calculate_class_counts(file_prefix, num_files, save_dir):
    all_labels = []

    # Loop over all files to concatenate labels
    for file_index in range(num_files):
        label_file = os.path.join(save_dir, f"{file_prefix}_labels_{file_index}.pt")
        labels = torch.load(label_file)
        all_labels.append(labels)

    # Concatenate all labels into a single tensor
    all_labels_concatenated = torch.cat(all_labels, dim=0)

    # Perform bincount on the concatenated labels
    total_counts = torch.bincount(all_labels_concatenated)

    return total_counts


def extract_features2(model, dataloader, device, cfg, file_prefix, batches_per_file, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    features = []
    labels = []
    batch_count = 0
    file_index = 0

    with torch.no_grad():
        for data, label in tqdm(dataloader, desc="Extracting features"):
            inputs = data[cfg['modalities'][0]]
            inputs = inputs.to(device)
            feature = model(inputs.permute(0,2,1,3,4))
            features.append(feature[0].cpu())
            labels.append(label)

            batch_count += 1
            if batch_count >= batches_per_file:
                feature_file = os.path.join(save_dir, f"{file_prefix}_features_{file_index}.pt")
                label_file = os.path.join(save_dir, f"{file_prefix}_labels_{file_index}.pt")
                torch.save(torch.cat(features, dim=0), feature_file)
                torch.save(torch.cat(labels, dim=0), label_file)
                features = []
                labels = []
                file_index += 1
                batch_count = 0

        # Save remaining data if any
        if features:
            feature_file = os.path.join(save_dir, f"{file_prefix}_features_{file_index}.pt")
            label_file = os.path.join(save_dir, f"{file_prefix}_labels_{file_index}.pt")
            torch.save(torch.cat(features, dim=0), feature_file)
            torch.save(torch.cat(labels, dim=0), label_file)

         #Return the number of files created
    return file_index + 1 if features else file_index

def delete_feature_files(file_prefix, num_files, save_dir):
    for file_index in range(num_files):
        feature_file = os.path.join(save_dir, f"{file_prefix}_features_{file_index}.pt")
        label_file = os.path.join(save_dir, f"{file_prefix}_labels_{file_index}.pt")
        if os.path.exists(feature_file):
            os.remove(feature_file)
        if os.path.exists(label_file):
            os.remove(label_file)

def train_classifier2(classifier, file_prefix, num_files, num_val_files, criterion, optimizer, num_epochs, batch_size, device, save_dir):
    classifier.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for file_index in range(num_files):
            feature_file = os.path.join(save_dir, f"{file_prefix}_features_{file_index}.pt")
            label_file = os.path.join(save_dir, f"{file_prefix}_labels_{file_index}.pt")
            features = torch.load(feature_file)
            labels = torch.load(label_file)
            num_samples = features.size(0)
            num_batches = (num_samples + batch_size - 1) // batch_size

            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, num_samples)
                batch_features = features[start:end].to(device)
                batch_labels = labels[start:end].to(device)

                optimizer.zero_grad()
                outputs = classifier(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

        epoch_loss = total_loss / (num_batches * num_files)
        epoch_accuracy = 100 * total_correct / total_samples
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%')
        if epoch % 10 == 0:
            val_accuracy, balanced_val_acc = evaluate_classifier2(classifier, 'val', num_val_files, batch_size, device, save_dir)
            


def evaluate_classifier2(classifier, file_prefix, num_files, batch_size, device, save_dir, mode='test'):
    classifier.eval()
    all_predictions = []
    all_labels = []

    for file_index in range(num_files):
        feature_file = os.path.join(save_dir, f"{file_prefix}_features_{file_index}.pt")
        label_file = os.path.join(save_dir, f"{file_prefix}_labels_{file_index}.pt")
        features = torch.load(feature_file)
        labels = torch.load(label_file)
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
    if mode == 'test':
        logging.info(f'Test Accuracy: {accuracy}%, Balanced Test Accuracy: {balanced_acc}%')
    elif mode == 'val':
        logging.info(f'Validation Accuracy: {accuracy}%, Balanced Validation Accuracy: {balanced_acc}%')
    return accuracy, balanced_acc



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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
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