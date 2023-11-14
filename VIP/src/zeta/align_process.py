import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from info_nce_pytorch import InfoNCE
import logging

def align_modalities_process(multi_modality_model, train_loader, val_loader, num_epochs=10, learning_rate=0.0001, temperature=0.1):
    """
    Train and validate a multi-modality model.

    :param multi_modality_model: The multi-modality model to be trained.
    :param train_loader: DataLoader for the training data.
    :param val_loader: DataLoader for the validation data.
    :param modalities_encoders: Dictionary of modality encoders.
    :param num_epochs: Number of epochs for training.
    :param learning_rate: Learning rate for the optimizer.
    :param temperature: Temperature parameter for InfoNCE loss.
    """

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(multi_modality_model.parameters(), lr=learning_rate)
    info_nce_loss = InfoNCE(temperature=temperature, reduction='mean', negative_mode='paired')

    # Placeholder for best validation loss
    best_val_loss = float('inf')

    logging.info("Starting training loop")

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        multi_modality_model.train()
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Training")

        for batch_data, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            embeddings = {}
            for modality in batch_data.keys():
                if modality in multi_modality_model.modalities_encoders:
                    data = batch_data[modality].cuda()
                    embeddings[modality] = multi_modality_model.module.forward_encoder(modality, data)

            modality_keys = list(embeddings.keys())
            loss = info_nce_loss(embeddings[modality_keys[0]], embeddings[modality_keys[1]])

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss / len(train_loader):.4f}")

        # Validation loop
        multi_modality_model.eval()
        val_loss = 0.0
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Validation")

        with torch.no_grad():
            for batch_data, _, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                embeddings = {}
                for modality in batch_data.keys():
                    if modality in multi_modality_model.modalities_encoders:
                        data = batch_data[modality].cuda()
                        embeddings[modality] = multi_modality_model.module.forward_encoder(modality, data)

                modality_keys = list(embeddings.keys())
                loss = info_nce_loss(embeddings[modality_keys[0]], embeddings[modality_keys[1]])
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(multi_modality_model.state_dict(), 'best_multi_modality_model.pth')
                logging.info("Best model saved")

    logging.info("Training complete!")