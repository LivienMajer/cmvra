import torch
import numpy as np
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data(data, modality, config):
    """
    Apply preprocessing specific to different encoder models as defined in the configuration.
    Depending on the configuration, handle single or mixed encoder setups appropriately.
    """
    if config['encoder_model'] == 'MIX':
        encoder = config['modalities_encoders'].get(modality, 'default')
    else:
        encoder = config['encoder_model']
    if encoder == 'MAE':
        return data.permute(0, 2, 1, 3, 4)
    elif encoder == 'OMNIVORE':
        return torch.cat((data, data[:, :, 0:1, :, :]), 2).permute(0, 2, 1, 3, 4)
    elif encoder == 'MAEPS':
        return data.view(data.size(0), data.size(1), -1)
    return data

def gather_embeddings_and_labels(data_loader, visual_model, device, config):
    embeddings = {modality: [] for modality in visual_model.module.modalities_encoders.keys()}
    labels_list = []
    for batch_data, labels in tqdm(data_loader, desc="Gathering data"):
        labels = labels.to(device)
        for modality, data in batch_data.items():
            if modality in visual_model.module.modalities_encoders:
                data = preprocess_data(data.to(device), modality, config)
                encoder_output = visual_model.module.forward_encoder(modality, data)
                embeddings[modality].append(encoder_output.detach().cpu())
        labels_list.append(labels.cpu())
    for modality in embeddings:
        embeddings[modality] = torch.cat(embeddings[modality], dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    return embeddings, labels



def cosine_distance(x, y):
    return 1 - cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))

@torch.no_grad()
def eval_knn(visual_model, train_loader, test_loader, device, config, k=5):
    visual_model.eval()
    logging.info("Processing training data...")
    train_embeddings, train_labels = gather_embeddings_and_labels(train_loader, visual_model, device, config)
    logging.info("Processing test data...")
    test_embeddings, test_labels = gather_embeddings_and_labels(test_loader, visual_model, device, config)
    accuracies = {}
    with open('/home/bas06400/daa/activity_mapping.txt', 'r') as f:
        activity_mapping = f.readlines()
    activity_mapping = [line.strip().split(': ')[1] for line in activity_mapping]

    for modality in train_embeddings:
        knn = KNeighborsClassifier(n_neighbors=k, metric=cosine_distance)
        knn.fit(train_embeddings[modality], train_labels)
        test_predictions = knn.predict(test_embeddings[modality])
        accuracy = accuracy_score(test_labels, test_predictions)
        balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
        cm = confusion_matrix(test_labels, test_predictions)
        accuracies[modality] = {
            'Standard Accuracy': accuracy,
            'Balanced Accuracy': balanced_acc,
            'Confusion Matrix': cm
        }
        logging.info(f"Modality: {modality}, kNN Standard Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_acc:.4f}")

        # Plot and save the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(pd.DataFrame(cm, index=activity_mapping, columns=activity_mapping), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {modality}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plot_filename = f'/home/bas06400/Thesis/VIP/src/plots/confusion_matrix_{modality}.png'
        plt.savefig(plot_filename)
        plt.close()
        logging.info(f"Confusion Matrix plot for {modality} saved to {plot_filename}")

    return accuracies