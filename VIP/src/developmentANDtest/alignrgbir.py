from modeling.VidCLIP import VidCLIP
from easydict import EasyDict as edict
import torch
import getpass
import sys
import os
from zeta.loss import InfoNCELoss1, SigmoidContrastiveMultiModalLoss

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
    
    
    # Initialize empty lists for each modality in the first sample
    for modality in batch[0][0].keys():
        collated_data[modality] = []
    
    for data, label  in batch:
        collated_labels.append(label-1)
        for modality, frames in data.items():
            collated_data[modality].append(frames)
        
    # Convert lists to tensors for each modality
    for modality, frames_list in collated_data.items():
        collated_data[modality] = torch.stack(frames_list)
    
    collated_labels = torch.tensor(collated_labels)
    
    return collated_data, collated_labels


# Create a DataLoader
batch_size = 32 #16
shuffle = True
num_workers = 20
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
ir_model.load_state_dict(state_dict)

ir_model.vision_model.embeddings.patch_embedding = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16), bias=False)

import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from info_nce_pytorch import InfoNCE
print('count',torch.cuda.device_count())
model = model.cuda(0)
ir_model = ir_model.cuda(0)
model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])  # Assuming GPUs 3 and 4 are available
ir_model = torch.nn.DataParallel(ir_model, device_ids=[0,1,2,3])
#for name, param in model.module.named_parameters():
 #       print(name, param.device)
# Hyperparameters
learning_rate = 0.0001
num_epochs = 10
temperature = 0.1  # Temperature parameter for InfoNCE loss

# Initialize the optimizer
optimizer = optim.Adam(ir_model.parameters(), lr=learning_rate)

info_nce_loss = SigmoidContrastiveMultiModalLoss()#InfoNCELoss1(temperature=temperature)#InfoNCE(temperature=temperature, reduction='mean', negative_mode='paired')

# Placeholder for best validation loss
best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    model.train()
    ir_model.train()
    # Wrap dataloader with tqdm for progress bar
    for batch_data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move data to GPU
        rgb_data = batch_data['rgb'].cuda() #.to('cuda:3')
        ir_data = batch_data['ir'].cuda() #.to('cuda:3')

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: Get embeddings or representations from model
        rgb_emb  = model(rgb_data)
        ir_emb  = ir_model(ir_data)

        loss = info_nce_loss(*[F.normalize(rgb_emb, p=2, dim=1),F.normalize(ir_emb, p=2, dim=1)])
        # Compute the contrastive loss
        #loss = info_nce_loss(ir_emb, rgb_emb)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()
        #break
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss / len(train_loader):.4f}")
    # Validation loop
    model.eval()
    ir_model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch_data, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            rgb_data = batch_data['rgb'].cuda() #.to('cuda:3')
            ir_data = batch_data['ir'].cuda() #.to('cuda:3')
            rgb_emb  = model(rgb_data)
            ir_emb  = ir_model(ir_data)
            loss = info_nce_loss(*[F.normalize(rgb_emb, p=2, dim=1),F.normalize(ir_emb, p=2, dim=1)])
            #loss = info_nce_loss(ir_emb, rgb_emb)
            val_loss += loss.item()
            #break
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
        
        # Save the best model (optional)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ir_model.state_dict(), 'best_ir_encoder4.pth')

print("Training complete!")