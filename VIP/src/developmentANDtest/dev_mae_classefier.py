import logging
from functools import partial
from omnimae import make_conv_or_linear, reshape_and_init_as_mlp
import torch
import torch.nn as nn
from omnimae import vit_base_mae_finetune_ssv2, vit_base_mae_pretraining, vit_large_mae_finetune_ssv2, vit_base_mae_finetune_in1k, make_conv_or_linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from vision_transformer import (
    Attention,
    Decoder,
    PadIm2Video,
    VisionTransformer,
)
from omegaconf import OmegaConf
from timm.models.layers import trunc_normal_
from torch.hub import load_state_dict_from_url
import random
from torch import optim

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def load_decoder_model(decoder_path, model):
    model.load_state_dict(torch.load(decoder_path),strict=False)
    return model

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, label in tqdm(dataloader, desc="Extracting features"):
            inputs = data['rgb']# Adjust according to your data format
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
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {running_loss / num_batches}")
            evaluate_classifier(classifier, val_features, val_labels, batch_size, device)

def evaluate_classifier(classifier, features, labels, batch_size, device):
    classifier.eval()
    correct = 0
    total = 0
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
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test or Val Accuracy: {accuracy}%')
    return accuracy

# Load your saved model
decoder_path = '/home/bas06400/Thesis/VIP/src/align_checkpoints/mae_checkpoints/checkpoint_rgb_0_20240120_220946.pth' # update this path
my_model = VisionTransformer(
        img_size=[3, 16, 224, 224],
        patch_size=[2, 16, 16],
        in_chans=3,
        embed_dim=768,
        depth=12,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=12,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_path_type="progressive",
        classifier_feature="global_pool",
        use_cls_token=False,
        learnable_pos_embed=False,
        layer_scale_type=None,
        layer_scale_init_value=0.1,
        patch_embed_type="generic",
        patch_embed_params_list=[
            PadIm2Video(ntimes=2, pad_type="repeat"),
            make_conv_or_linear(
                layer=torch.nn.Conv3d(
                    in_channels=3,
                    kernel_size=[2, 16, 16],
                    out_channels=768,
                    stride=[2, 16, 16],
                ),
                init_weight=partial(reshape_and_init_as_mlp),
            ),
        ],
        layer_norm_eps=1e-6,
        masked_image_modeling=False,
        patch_drop_max_patches=-1,
        add_pos_same_dtype=False,
        patch_dropping=True,
        post_encoder_params=None,
        decoder=None,
        mask_token_embed_dim=None,
        )
my_model = load_decoder_model(decoder_path, my_model)

import sys
sys.path.append("/home/bas06400/Thesis/VIP/src/")

from zeta.data_loader import load_dataloaders

data_root = "/net/polaris/storage/deeplearning/ntu"
modalities = ["rgb", "ir", "depth"]
batch_size = 8
num_workers = 10
pin_memory = True
split = "0"
random_sample = False
config = {'dataset':'DAA'}
train_data, val_data, test_data = load_dataloaders(data_root=data_root,
                                                    modalities=modalities,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory,
                                                        split=split,
                                                        random_sample=random_sample,
                                                        config=config)
# Define optimizer and scheduler
my_model = my_model.to('cuda:3')
# Extract features using your dataloaders (train and val)
train_features, train_labels = extract_features(my_model, train_data, device='cuda:3')
val_features, val_labels = extract_features(my_model, val_data, device='cuda:3')
test_features, test_labels = extract_features(my_model, test_data, device='cuda:3')
# Define and train the linear classifier
input_dim = train_features.shape[1] # Adjust based on your feature size
num_classes = 34 # Adjust based on your number of classes
linear_classifier = LinearClassifier(768, num_classes).to('cuda:3')

class_counts = torch.bincount(train_labels)
class_weights = 1. / class_counts
class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1

# Move weights to the correct device
class_weights = class_weights.to('cuda:3')

# Create the weighted loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(linear_classifier.parameters(), lr=0.001)
num_epochs = 300 # Adjust as necessary

train_classifier(linear_classifier, train_features, val_features, train_labels, val_labels, criterion, optimizer, num_epochs, batch_size=64, device='cuda:3')

# Evaluate the classifier
evaluate_classifier(linear_classifier, test_features, test_labels, batch_size=64, device='cuda:3')
