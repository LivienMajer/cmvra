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


    

def generate_random_mask(batch_size, num_patches, mask_ratio):
    """
    Generates a random mask for the patches.
    """
    num_masked = int(mask_ratio * num_patches)
    mask = torch.zeros((batch_size, num_patches), dtype=torch.bool)
    for i in range(batch_size):
        masked_indices = random.sample(range(num_patches), num_masked)
        mask[i, masked_indices] = True
    return mask

class MultiModalReconstructionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = VisionTransformer(
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
        masked_image_modeling=True,
        patch_drop_max_patches=-1,
        add_pos_same_dtype=False,
        patch_dropping=True,
        post_encoder_params=None,
        decoder=None,
        mask_token_embed_dim=None,
        )

        # Decoders for each modality
        self.ir_decoder = Decoder(
            first_patch_idx=0,
            patches_layout=self.rgb_encoder.patch_embed.patches_layout,
            attn_target=partial(Attention, num_heads=16),
            decoder_depth=4,
            decoder_embed_dim=384,
            embed_dim=768,
            learnable_pos_embed=False,
            qkv_bias=True,
        )
        self.rgb_decoder = Decoder(
            first_patch_idx=0,
            patches_layout=self.rgb_encoder.patch_embed.patches_layout,
            attn_target=partial(Attention, num_heads=16),
            decoder_depth=4,
            decoder_embed_dim=384,
            embed_dim=768,
            learnable_pos_embed=False,
            qkv_bias=True,
        )
        self.depth_decoder = Decoder(
            first_patch_idx=0,
            patches_layout=self.rgb_encoder.patch_embed.patches_layout,
            attn_target=partial(Attention, num_heads=16),
            decoder_depth=4,
            decoder_embed_dim=384,
            embed_dim=768,
            learnable_pos_embed=False,
            qkv_bias=True,
        )
        self.ir_head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=384, out_features=1536),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=0.02),
        )
        self.rgb_head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=384, out_features=1536),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=0.02),
        )
        self.depth_head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=384, out_features=1536),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=0.02),
        )
        self.norm = nn.LayerNorm(384)
    def forward(self, rgb_input, mask):
        
        encoded_features, input_shape, pos_embed = self.rgb_encoder(rgb_input, mask=mask)
        #print(encoded_features[1].shape)
        # Decoding for each modality
        ir_reconstruction = self.ir_head(self.norm(self.ir_decoder(encoded_features, input_shape, pos_embed)))
        rgb_reconstruction = self.rgb_head(self.norm(self.rgb_decoder(encoded_features, input_shape, pos_embed)))
        depth_reconstruction = self.depth_head(self.norm(self.depth_decoder(encoded_features, input_shape, pos_embed)))

        return ir_reconstruction, rgb_reconstruction, depth_reconstruction

# Loss function
def mse_loss(reconstructed, original):
    criterion = nn.MSELoss()
    loss = criterion(reconstructed, original)
    return loss

import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to save the RGB encoder weights
def save_best_model(model, best_val_loss, current_val_loss, epoch, save_dir='model_weights', filename='ir_encoder_20best08.pth'):
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filename)
        torch.save(model.module.rgb_encoder.state_dict(), save_path)
        logger.info(f"Saved new best RGB encoder model at epoch {epoch} to {save_path}")
    return best_val_loss

def train(model, train_loader, optimizer, scheduler, num_epochs, val_loader, device='cuda:2'):
    model.train()
    num_patches = 1568
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0
        losses_per_modality = {'ir': 0.0, 'rgb': 0.0, 'depth': 0.0}
        for batch_data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            rgb_input = batch_data['rgb'].to(device)
            optimizer.zero_grad()

            batch_size = rgb_input.size(0)
            mask = generate_random_mask(batch_size, num_patches, 0.8).to(device)
            #print(f"This is the mask shape: {mask.shape}")
            ir_reconstructed, rgb_reconstructed, depth_reconstructed = model(rgb_input.permute(0,2,1,3,4), mask)

            # Calculate and log losses for each modality
            #loss_ir = mse_loss(ir_reconstructed.view(-1,3,16,224,224), batch_data['ir'].permute(0,2,1,3,4).to(device))
            loss_rgb = mse_loss(rgb_reconstructed.view(-1,3,16,224,224), batch_data['rgb'].permute(0,2,1,3,4).to(device))
            #loss_depth = mse_loss(depth_reconstructed.view(-1,3,16,224,224), batch_data['depth'].permute(0,2,1,3,4).to(device))
            loss = loss_rgb #+ loss_depth + loss_ir
            total_loss += loss.item()
            #losses_per_modality['ir'] += loss_ir.item()
            losses_per_modality['rgb'] += loss_rgb.item()
            #losses_per_modality['depth'] += loss_depth.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        avg_losses_modality = {k: v / len(train_loader) for k, v in losses_per_modality.items()}

        # Log the losses for each modality
        logger.info(f"Epoch: {epoch}, Average Training Loss: {avg_loss}, Losses per Modality: {avg_losses_modality}")

        # Validation and learning rate scheduling
        if val_loader:
            val_loss, val_losses_modality = validate(model, val_loader, device)
            scheduler.step(val_loss)
            best_val_loss = save_best_model(model, best_val_loss, val_loss, epoch)
            logger.info(f"Epoch: {epoch}, Validation Loss: {val_loss}, Validation Losses per Modality: {val_losses_modality}")
        else:
            scheduler.step(avg_loss)
            best_val_loss = save_best_model(model, best_val_loss, avg_loss, epoch)


def validate(model, val_loader, device='cuda:2'):
    model.eval()
    num_patches = 1568
    total_val_loss = 0
    val_losses_per_modality = {'ir': 0.0, 'rgb': 0.0, 'depth': 0.0}
    val_progress_bar = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch_data, _ in val_progress_bar:
            rgb_input = batch_data['rgb'].to(device)

            batch_size = rgb_input.size(0)
            mask = generate_random_mask(batch_size, num_patches, 0.8).to(device)
            #print(f"This is the mask shape: {mask.shape}")
            ir_reconstructed, rgb_reconstructed, depth_reconstructed = model(rgb_input.permute(0,2,1,3,4), mask)

            #loss_ir = mse_loss(ir_reconstructed.view(-1,3,16,224,224), batch_data['ir'].permute(0,2,1,3,4).to(device))
            loss_rgb = mse_loss(rgb_reconstructed.view(-1,3,16,224,224), batch_data['rgb'].permute(0,2,1,3,4).to(device))
            #loss_depth = mse_loss(depth_reconstructed.view(-1,3,16,224,224), batch_data['depth'].permute(0,2,1,3,4).to(device))

            val_loss =  loss_rgb.item() # +loss_ir.item() + loss_depth.item()
            total_val_loss += val_loss
            #val_losses_per_modality['ir'] += loss_ir.item()
            val_losses_per_modality['rgb'] += loss_rgb.item()
            #val_losses_per_modality['depth'] += loss_depth.item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_losses_modality = {k: v / len(val_loader) for k, v in val_losses_per_modality.items()}

    return avg_val_loss, avg_val_losses_modality


if __name__ == "__main__":
    pretrained_model = vit_base_mae_pretraining(pretrained=True)
    pretrained_state_dict = pretrained_model.state_dict()
    del pretrained_model

    my_model = MultiModalReconstructionModel()

    # Loading VisionTransformer weights
    for key, value in pretrained_state_dict.items():
        if key.startswith('trunk.') and not key.startswith('trunk.decoder.'):
            new_key = key.replace('trunk.', 'rgb_encoder.')  
            if new_key in my_model.state_dict():
                my_model.state_dict()[new_key].copy_(value)

    # Loading Decoder weights
    for key, value in pretrained_state_dict.items():
        if key.startswith('trunk.decoder.'):
            for decoder_name in ['ir_decoder', 'rgb_decoder', 'depth_decoder']:
                new_key = key.replace('trunk.decoder.', decoder_name + '.')  
                if new_key in my_model.state_dict():
                    my_model.state_dict()[new_key].copy_(value)
    my_model.to('cuda:2')

    my_model = nn.DataParallel(my_model, device_ids= [ 2,3])
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
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Number of training epochs
    num_epochs = 100

    # Start training
    train(my_model, train_data, optimizer, scheduler, num_epochs, val_data)

 