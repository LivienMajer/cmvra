from modeling.VidCLIP import VidCLIP
from easydict import EasyDict as edict
import torch
import getpass
import sys
import os
import gc  

def setup_ccname():
    user = getpass.getuser()
    # check if k5start is running, exit otherwise
    try:
        pid = open("/tmp/k5pid_" + user).read().strip()
        os.kill(int(pid), 0)
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nk5start not running!\n")
        sys.exit(1)
    try:
        ccname = open("/tmp/kccache_" + user).read().split("=")[1].strip()
        os.environ['KRB5CCNAME'] = ccname
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nmaybe k5start not running?\n")
        sys.exit(1)

# Create an 'args' object from the provided JSON structure
args = edict({
    "clip_config": "openai/clip-vit-base-patch16",
    "clip_weights": "openai/clip-vit-base-patch16",
    "clip_vision_additional_config": edict({
        "type": "ViP",
        "temporal_size": 12,
        "if_use_temporal_embed": True,
        "logit_scale_init_value": 4.60,
        "add_cls_num": 3
    }),
    "e2e_weights_path": "path/to/CLIP-ViP-B/16/checkpoint"
})

# Initialize the model instance
model_instance = VidCLIP(args)
print(model_instance)

ckpt = torch.load('/home/bas06400/Thesis/pretrain_clipvip_base_16.pt')
model_instance.load_state_dict(ckpt)

from modeling.CLIP_ViP import CLIPVisionModel, CLIPVisionTransformer
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPPreTrainedModel
from torch import nn


# Given json_config
json_config = {
    # ... (rest of your JSON config)
    "additional_vision_config": {
        "type": "ViP",
        "temporal_size": 12,
        "if_use_temporal_embed": 1,
        "logit_scale_init_value": 4.60,
        "add_cls_num": 3,
        "hiiden_size": 12
    },
    # ... (rest of your JSON config)
}
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    
# Load the base CLIPConfig
clipconfig = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch16")

additional_vision_config_obj = SimpleNamespace(**json_config["additional_vision_config"])
setattr(clipconfig, "additional_vision_config", additional_vision_config_obj)
class CLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        # Pass the additional_vision_config to CLIPVisionTransformer
        self.vision_model = CLIPVisionTransformer(config, config.additional_vision_config)
        # Add the visual projection layer
        self.visual_projection = nn.Linear(768, 512, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Get the output from the vision_model
        vision_output = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        pooled_output = vision_output[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)
        
        return image_features

model = CLIPVisionModel(clipconfig)

# Assuming vidclip_model is the instance of your VidCLIP model
vidclip_model_weights = model_instance.state_dict()



# Prepare a dictionary to hold the relevant weights
state_dict = {}

# Copy weights from the vidclip_model_weights to state_dict
for name, param in vidclip_model_weights.items():
    if "vision_model" in name:
        new_name = name.replace("clipmodel.", "")  # remove the prefix
        state_dict[new_name] = param
    if "visual_projection" in name:
        new_name = name.replace("clipmodel.", "")  # remove the prefix
        state_dict[new_name] = param

# Load the state_dict into clip_vision_model
model.load_state_dict(state_dict)

for param in model.parameters():
    param.requires_grad_(False)

from multimodal_dataset import MultiModalVideoDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import torch
from torch.utils.data import random_split
import random

# Set the seed for reproducibility
seed = 42
random.seed(seed)  # Seed for Python's random module
torch.manual_seed(seed)  # Seed for PyTorch random number generators

data_root = '/net/polaris/storage/deeplearning/ntu'
data_list = '/home/bas06400/Thesis/rgb_ir_dataset.txt'
data = MultiModalVideoDataset(data_list, data_root, ['rgb','ir'], use_advanced_processing=True)

print(data[0][0]['rgb'].shape, data[0][0]['ir'].shape, data[0][1])

# Calculate lengths of splits
total_len = len(data)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len

# Split the dataset
train_data, val_data, test_data = random_split(data, [train_len, val_len, test_len])


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of data from MultiModalVideoDataset.
    
    Args:
    - batch (list): List of samples fetched from `MultiModalVideoDataset`.
    
    Returns:
    - collated_data (dict): Collated data for each modality.
    - collated_labels (tensor): Collated labels.
    """
    collated_data = {}
    collated_labels = []
    collated_idx =[]
    
    # Initialize empty lists for each modality in the first sample
    for modality in batch[0][0].keys():
        collated_data[modality] = []
    
    for data, label , idx in batch:
        collated_labels.append(label-1)
        for modality, frames in data.items():
            collated_data[modality].append(frames)
        collated_idx.append(idx)
    # Convert lists to tensors for each modality
    for modality, frames_list in collated_data.items():
        collated_data[modality] = torch.stack(frames_list)
    
    collated_labels = torch.tensor(collated_labels)
    
    return collated_data, collated_labels, collated_idx


# Create a DataLoader
batch_size = 16
shuffle = True
num_workers = 10
pin_memory = True

# Create a DataLoader for the training set
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=custom_collate_fn
)

# Create a DataLoader for the validation set
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,  
    shuffle=False,  
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=custom_collate_fn
)

# Create a DataLoader for the test set
test_loader = DataLoader(
    test_data,
    batch_size=batch_size,  
    shuffle=False,  
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=custom_collate_fn
)
"""
for batch_data, batch_labels in test_loader:
    
    print(batch_data['rgb'].shape,batch_data['ir'].shape)
    break
"""

ir_model = CLIPVisionModel(clipconfig)
# Load the state_dict into clip_vision_model
ir_model.vision_model.embeddings.patch_embedding = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16), bias=False)

ckpt_ir = torch.load('/home/bas06400/Thesis/VIP/src/best_ir_encoder2.pth')
# Remove the 'module.' prefix if it exists
new_state_dict = {}
for k, v in ckpt_ir.items():
    if k.startswith('module.'):
        # Remove the prefix
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

# Load the state dict into the model
ir_model.load_state_dict(new_state_dict)



for param in ir_model.parameters():
    param.requires_grad_(False)

import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from info_nce_pytorch import InfoNCE



class VIPxClassefier(nn.Module):
    def __init__(self,encoder1=None, encoder2=None):
        super(VIPxClassefier, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2

        self.rgb_classefier = nn.Linear(512, 60)
        self.ir_classefier = nn.Linear(512, 60)


    def forward(self, image, text):
        
        rgb_out, ir_out = self.encoder1(image), self.encoder2(text)
        
        # Project the 400D embeddings to 512D
        rgb_features = self.rgb_classefier(rgb_out)
        ir_features = self.ir_classefier(ir_out)
        return rgb_features, ir_features


import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


num_epochs = 10
model = VIPxClassefier(encoder1=model, encoder2=ir_model)
model = model.to('cuda:3')
# Cross-Entropy Loss function
criterion = torch.nn.CrossEntropyLoss()

# Hyperparameters
learning_rate = 0.0001

# Placeholder for best validation loss
best_val_loss = float('inf')

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

# Function to compute accuracy
def compute_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / len(labels)

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Training loop with modified loss and accuracy calculations
for epoch in range(num_epochs):
    epoch_loss = 0.0
    total_accuracy_rgb = 0.0
    total_accuracy_ir = 0.0
    
    model.train()
    
    for batch_data, batch_labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move data and labels to GPU
        rgb_data = batch_data['rgb'].to('cuda:3')
        ir_data = batch_data['ir'].to('cuda:3')
        labels = batch_labels.to('cuda:3')

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        rgb_logits, ir_logits = model(rgb_data, ir_data)
        
        # Compute the losses
        loss_rgb = criterion(rgb_logits, labels)
        loss_ir = criterion(ir_logits, labels)
        loss = (loss_rgb + loss_ir) / 2  # Average the two losses

        # Compute accuracies
        accuracy_rgb = compute_accuracy(rgb_logits, labels)
        accuracy_ir = compute_accuracy(ir_logits, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()
        total_accuracy_rgb += accuracy_rgb
        total_accuracy_ir += accuracy_ir

        # Memory cleanup
        del rgb_data, ir_data, labels, rgb_logits, ir_logits
        clear_memory()

    avg_accuracy_rgb = total_accuracy_rgb / len(train_loader)
    avg_accuracy_ir = total_accuracy_ir / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, RGB Accuracy: {avg_accuracy_rgb:.4f}, IR Accuracy: {avg_accuracy_ir:.4f}")
    
    
    clear_memory()
    # Validation loop
    model.eval()
    total_val_accuracy_rgb = 0.0
    total_val_accuracy_ir = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for batch_data, batch_labels, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            rgb_data = batch_data['rgb'].to('cuda:3')
            ir_data = batch_data['ir'].to('cuda:3')
            labels = batch_labels.to('cuda:3')
            
            rgb_logits, ir_logits = model(rgb_data, ir_data)
            
            # Compute the losses
            loss_rgb = criterion(rgb_logits, labels)
            loss_ir = criterion(ir_logits, labels)
            loss = (loss_rgb + loss_ir) / 2
            
            # Compute accuracies
            accuracy_rgb = compute_accuracy(rgb_logits, labels)
            accuracy_ir = compute_accuracy(ir_logits, labels)

            val_loss += loss.item()
            total_val_accuracy_rgb += accuracy_rgb
            total_val_accuracy_ir += accuracy_ir

            # Memory cleanup
            del rgb_data, ir_data, labels, rgb_logits, ir_logits
            clear_memory()

    avg_val_accuracy_rgb = total_val_accuracy_rgb / len(val_loader)
    avg_val_accuracy_ir = total_val_accuracy_ir / len(val_loader)
    print(f"Validation Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss / len(val_loader):.4f}, RGB Accuracy: {avg_val_accuracy_rgb:.4f}, IR Accuracy: {avg_val_accuracy_ir:.4f}")
    clear_memory()
    # Save the best model (optional)
    avg_val_loss = val_loss / len(val_loader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_classefier_model.pth')

print("Training complete!")