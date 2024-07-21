"""
contains code from with the followoing sources with some adjustments for our specific use cases

CLIP-ViP: Video-and-Image-Pre-training
   Authors:  Hongwei Xue*, Yuchong Sun*, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, Jiebo Luo.
   Repository: https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP
   

OmniMAE: Omnivore Masked Autoencoder
   Author: facebookresearch
   Repository: https://github.com/facebookresearch/omnivore/tree/main/omnimae
"""


import json
import torch
import os
import logging
from easydict import EasyDict as edict
from modeling.CLIP_ViP import CLIPVisionModel, CLIPVisionTransformer, CLIPTextModel, CLIPTextTransformer
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPVisionConfig, CLIPTextConfig
from transformers import CLIPPreTrainedModel
from torch import nn
from modeling.VidCLIP import VidCLIP
from typing import Any, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from modeling.omnimae import make_conv_or_linear, reshape_and_init_as_mlp, vit_base_mae_pretraining 
from modeling.vision_transformer import (
    Attention,
    Decoder,
    PadIm2Video,
    VisionTransformer,
)
from modeling.omnivore import omnivore_swinB_imagenet21k
from modeling.MaeSkeletonPretrained import TransformerModel
from functools import partial
from timm.models.layers import trunc_normal_

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def adjust_state_dict_for_finetuning(state_dict):
    """ 
    Adjusts the state dictionary by removing 'module.' prefix from keys. Can also be done by setting strict option to false in load state dict
    but applying this can be used as a sanity check.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Remove 'module.' from key
        else:
            new_state_dict[k] = v
    return new_state_dict        

def initialize_vip_encoder(config, modality='rgb', freeze=True):
    """
    Initializes a VIP encoder based on the given configuration.

    Args:
        config (dict): Configuration for the VIP encoder.
        modality (str): Input modality (default: 'rgb').
        freeze (bool): Whether to freeze model parameters.

    Returns:
        CLIPVisionModel: Initialized VIP encoder.
    """

    # Create an 'args' object from the configuration
    args = edict({
        "clip_config": config["clip_config"],
        "clip_weights": config["clip_weights"],
        "clip_vision_additional_config": edict(config["clip_vision_additional_config"]),
        "e2e_weights_path": "path/to/CLIP-ViP-B/16/checkpoint"
    })

    # Initialize the model instance
    model_instance = VidCLIP(args)
    

    # Load model weights
    ckpt = torch.load(config["e2e_weights_path"])
        # Check if the model is fine-tuned and adjust state dictionary if necessary
    #if config.get("is_finetuned", False):
     #   ckpt = adjust_state_dict_for_finetuning(ckpt)

    try:
        model_instance.load_state_dict(ckpt)
    except RuntimeError as e:
        if "Missing key(s) in state_dict" in str(e):
            # Retry with adjusted state dictionary
            print("Adjusting state dictionary for fine-tuned model.")
            ckpt = adjust_state_dict_for_finetuning(ckpt)
            model_instance.load_state_dict(ckpt)
        else:
            raise e  # Reraise the exception if it's a different error

    # Load the base CLIPConfig
    clipconfig = CLIPVisionConfig.from_pretrained(args.clip_config)

    additional_vision_config_obj = SimpleNamespace(**config["clip_vision_additional_config"])
    setattr(clipconfig, "additional_vision_config", additional_vision_config_obj)

    # Initialize the CLIPVisionModel
    model = CLIPVisionModel(clipconfig)

    # Prepare the state_dict for loading
    state_dict = {}
    vidclip_model_weights = model_instance.state_dict()
    for name, param in vidclip_model_weights.items():
        if "vision_model" in name or "visual_projection" in name:
            new_name = name.replace("clipmodel.", "")  # remove the prefix
            state_dict[new_name] = param

    
    model.load_state_dict(state_dict)

    if freeze:
        # Freeze the parameters if required
        for param in model.parameters():
            param.requires_grad_(False)
    else:
        # Freeze the parameters if required
        for param in model.parameters():
            param.requires_grad_(True)

    # adjust channel dim in patch projection layer
    #if modality in ['ir', 'depth', 'skeleton']:
      #  model.vision_model.embeddings.patch_embedding = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16), bias=False)

    if modality in ['skeleton']:
        # Modify patch embedding for 1D skeleton input (1 channel, 3 coordinates per joint)
        model.vision_model.embeddings.patch_embedding = nn.Conv2d(1, 768, kernel_size=(1, 3), stride=(1, 3), bias=False)
        # Overwrite position embedding for 25 body joints + 1 cls token (total 26)
        model.vision_model.embeddings.position_embedding = nn.Embedding(26, 768)
        # Overwrite position IDs for 25 joints + 1 cls token
        model.vision_model.embeddings.register_buffer("position_ids", torch.arange(26).expand((1, -1)))

    return model


def initalize_aligned_encoder(config_path, weights_path, modality='rgb', freeze=True, add_classefier=True):

     # Load configuration
    with open(config_path) as f:
        config = json.load(f)

    # Create an 'args' object from the configuration
    args = edict({
        "clip_config": config["clip_config"],
        "clip_weights": config["clip_weights"],
        "clip_vision_additional_config": edict(config["clip_vision_additional_config"]),
        "e2e_weights_path": "path/to/CLIP-ViP-B/16/checkpoint"
    })

    # Load the base CLIPConfig
    clipconfig = CLIPVisionConfig.from_pretrained(args.clip_config)

    additional_vision_config_obj = SimpleNamespace(**config["additional_vision_config"])
    setattr(clipconfig, "additional_vision_config", additional_vision_config_obj)

    # Initialize the CLIPVisionModel
    model = CLIPVisionModel(clipconfig)


def initialize_vip_text_encoder(config, device):
    ckpt = torch.load(config["e2e_weights_path"], map_location=device)

    # Prepare a dictionary to hold the relevant weights
    state_dict = {}
    
    # Copy weights from the vidclip_model_weights to state_dict
    for name, param in ckpt.items():
        if "text_model" in name or "text_projection" in name:
            new_name = name.replace("clipmodel.", "")  # remove the prefix
            state_dict[new_name] = param
    #print(state_dict)
    # Initialize your custom CLIP text model
    clip_text_config = CLIPTextConfig.from_pretrained("openai/clip-vit-base-patch16")
    text_model = CustomCLIPTextModel(clip_text_config)  

    try:
        text_model.load_state_dict(state_dict)
    except RuntimeError as e:
        if "Missing key(s) in state_dict" in str(e):
            # Retry with adjusted state dictionary
            print("Adjusting state dictionary for fine-tuned model.")
            state_dict = adjust_state_dict_for_finetuning(state_dict)
            text_model.load_state_dict(state_dict)
        else:
            raise e  # Reraise the exception if it's a different error

    return text_model




class CustomCLIPTextModel(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        # No additional text config passed here as it's not provided
        self.text_model = CLIPTextTransformer(config)
        
        self.text_projection = nn.Linear(in_features=512, out_features=512, bias=False)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # Call the original forward method to get the model outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        # Apply the text_projection layer to the pooled_output (assuming you want to project the pooled output)
        projected_output = self.text_projection(outputs.pooler_output)
        
        if not return_dict:
            # If not returning a dict, convert the BaseModelOutputWithPooling to a tuple,
            # append the projected_output to the tuple, and return
            outputs_tuple = (
                outputs.last_hidden_state,
                projected_output,
                outputs.hidden_states,
                outputs.attentions
            )
            return outputs_tuple
        
        # Otherwise, create a new BaseModelOutputWithPooling containing the projected_output and return
        return BaseModelOutputWithPooling(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=projected_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



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
    

class MultiModalityModel(nn.Module):
    """
    Multi-modality model integrating different encoders and classifiers.

    Attributes:
        modalities_encoders (nn.ModuleDict): Encoders for different modalities.
        attention (nn.MultiheadAttention): For feature fusion.
        final_classifier (nn.Linear): For final classification after fusion.
    """
    def __init__(self, modalities_encoders, num_classes, in_features):
        super(MultiModalityModel, self).__init__()
        self.modalities_encoders = nn.ModuleDict(modalities_encoders)

        # Adding a linear layer for each modality
        for modality in modalities_encoders:
            setattr(self, f"{modality}_classifier", nn.Linear(in_features, num_classes))

        self.attention = nn.MultiheadAttention(in_features, num_heads=8, batch_first=True)
        self.final_classifier = nn.Linear(in_features, num_classes)

    def forward_encoder(self, modality, x):
        if modality in self.modalities_encoders:
            return self.modalities_encoders[modality](x)
        else:
            raise ValueError(f"Modality {modality} not recognized")

    def forward_classifier(self, modality, x):
        encoder_output = self.forward_encoder(modality, x)
        classifier = getattr(self, f"{modality}_classifier")
        return classifier(encoder_output)
    
    def forward_classifier_only(self, modality, x):
        classifier = getattr(self, f"{modality}_classifier")
        return classifier(x)
    
    def forward_fusion(self, x):
        # concatenated_features should be of shape (batch_size, num_modalities, in_features)
        
        # Applying attention directly on the concatenated features
        attention_output, _ = self.attention(x, x, x)
        
        # Reducing sequence dimension by taking the mean to get (batch_size, in_features)
        attention_output = attention_output.mean(dim=1)
        
        # Final classification
        return self.final_classifier(attention_output)
    

class MaeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VisionTransformer(
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
        self.decoder = Decoder(
            first_patch_idx=0,
            patches_layout=self.encoder.patch_embed.patches_layout,
            attn_target=partial(Attention, num_heads=16),
            decoder_depth=4,
            decoder_embed_dim=384,
            embed_dim=768,
            learnable_pos_embed=False,
            qkv_bias=True,
        )
        self.head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=384, out_features=1536),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=0.02),
        )
        
        self.norm = nn.LayerNorm(384)

    def forward(self, rgb_input, mask):
        encoded_features, input_shape, pos_embed = self.encoder(rgb_input, mask=mask)
        return self.head(self.norm(self.decoder(encoded_features, input_shape, pos_embed)))
    

def init_mae_model(gpus, config):
    pretrained_model = vit_base_mae_pretraining(pretrained=True)
    pretrained_state_dict = pretrained_model.state_dict()
    del pretrained_model

    model = MaeModel()
    # Loading VisionTransformer weights
    for key, value in pretrained_state_dict.items():
        if key.startswith('trunk.') and not key.startswith('trunk.decoder.'):
            new_key = key.replace('trunk.', 'encoder.')  
            if new_key in model.state_dict():
                model.state_dict()[new_key].copy_(value)

    # Loading Decoder weights
    for key, value in pretrained_state_dict.items():
        if key.startswith('trunk.decoder.'):
            new_key = key.replace('trunk.decoder.', 'decoder.')  #####do never forget to set a . dumb fuck 
            if new_key in model.state_dict():
                model.state_dict()[new_key].copy_(value)
    model = model.cuda(sorted(gpus)[0])
    
    return torch.nn.DataParallel(model, device_ids=sorted(gpus))

class MAEEncoderWithLinear(torch.nn.Module):
    def __init__(self, encoder, classifier):
        super(MAEEncoderWithLinear, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


def init_mae_encoder(cfg, checkpoint, device, return_class=True, freeze=False):
    """
    Initializes a MAE (Masked Autoencoder) encoder based on the given configuration. 

    Args:
        cfg (dict): Configuration dictionary.
        checkpoint (str): Path to the checkpoint file.
        device (torch.device): The device to load the model on.
        return_class (bool, optional): Whether to return the classifier. Defaults to True.
        freeze (bool, optional): Whether to freeze the encoder parameters. Defaults to False.

    Returns:
        Union[nn.Module, Tuple[nn.Module, nn.Module]]: The initialized MAE encoder (and classifier if return_class is True).
    """
    encoder = VisionTransformer(
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
    checkpoint_path = os.path.join(cfg['cktp_dir'], checkpoint)
    # Check if the checkpoint file exists
    if not os.path.isfile(checkpoint_path):
        logging.info(f"Checkpoint '{checkpoint}' not found in '{cfg['cktp_dir']}'.")
        # Set a default checkpoint directory or handle the missing file as needed
        default_cktp_dir = "/home/bas06400/Thesis/VIP/src/align_checkpoints/mae_checkpoints/"
        checkpoint_path = os.path.join(default_cktp_dir, checkpoint)
        # Optionally, check again if the checkpoint exists in the default directory and handle if it still doesn't exist
        if not os.path.isfile(checkpoint_path):
            logging.info(f"Default checkpoint '{checkpoint}' also not found. Please check your paths.")
            
            raise FileNotFoundError(f"Checkpoint '{checkpoint}' not found in both specified and default directories.")
        else:
            # Proceed with loading the checkpoint since it exists
            encoder.load_state_dict(torch.load(checkpoint_path, map_location=f'cuda:{device}')['model_state_dict'], strict=False)
            logging.info(f"Checkpoint {checkpoint} loaded succesfully")
    else:
        # Proceed with loading the checkpoint since it exists
        encoder.load_state_dict(torch.load(checkpoint_path, map_location=f'cuda:{device}')['model_state_dict'], strict=False)
        logging.info(f"Checkpoint {checkpoint} loaded succesfully")
    embbeding = LinearClassifier(cfg.get('input_dim', 768), 512 )
    encoder = MAEEncoderWithLinear(encoder, embbeding)
    # Freeze the encoder parameters if freeze is True
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False

    classifier = LinearClassifier(cfg.get('input_dim', 768), cfg['num_classes'], )
    if return_class == True:
        return encoder, classifier
    else:
        return encoder
    

def init_omnivore_encoder(cfg, device, freeze=False):
    model = omnivore_swinB_imagenet21k()
    
    model.heads = nn.Linear(1024, cfg['in_features'], bias=False) #overwrite datasetspecfic classification heads with a dimensionality adaption layer.
    if freeze:
        # Freeze all parameters in the model
        for param in model.parameters():
            param.requires_grad = False
        #print('we frooze')  
        # Unfreeze the parameters in the newly initialized head 
        for param in model.heads.parameters():
            param.requires_grad = True
            
    return model.to(f'cuda:{device}')

class DINOVforIR(nn.Module):
    """
    Adapts DINO v2 Vision Transformer for processing temporal IR (Infrared) data.

    This class modifies the original DINO v2 model to handle video frames by:
    1. Processing multiple frames in a single forward pass
    2. Applying mean pooling across the temporal dimension
    3. Adding a dimensionality reduction layer for the final embedding

    Attributes:
        vision_transformer (nn.Module): Pre-trained DINO v2 ViT-B/14 model
        dim_reduction (nn.Linear): Linear layer for reducing embedding dimension
        classifier (nn.Linear): Optional classifier layer (not used in forward method)

    Args:
        num_classes (int): Number of output classes (for classifier layer)
        embedding_dim (int): Dimension of the output embedding (default: 512)
        freeze_dino (bool): Whether to freeze the DINO v2 model parameters (default: False)
    """
    def __init__(self, num_classes, embedding_dim=512, freeze_dino=False):
        super(DINOVforIR, self).__init__()
        # Instantiate the DinoVisionTransformer model
        self.vision_transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # A linear layer for dimensionality reduction
        if freeze_dino:
            for param in self.vision_transformer.parameters():
                param.requires_grad = False
                
        self.dim_reduction = nn.Linear(768, embedding_dim)
        # Classifier layer
        self.classifier = nn.Linear(embedding_dim, num_classes)
        # Ensure the vision transformer model is in evaluation mode if not training
        self.vision_transformer.eval()

    def forward(self, video_frames):
        """
        video_frames: a tensor of shape (B, T, C, H, W)
        return_embedding: if True, return the clip embedding instead of the classification result
        """
        batch_size, num_frames, C, H, W = video_frames.size()
        # Reshape to process all frames at once
        video_frames = video_frames.view(batch_size * num_frames, C, H, W)
        # Compute frame embeddings
        frame_embeddings = self.vision_transformer(video_frames)
        # Reshape back to (B, T, embedding_dim)
        frame_embeddings = frame_embeddings.view(batch_size, num_frames, -1)
        # Mean pooling across frames
        clip_embedding = torch.mean(frame_embeddings, dim=1)
        # Dimensionality reduction
        clip_embedding = self.dim_reduction(clip_embedding)
        
        
        return clip_embedding
        
    
def init_dino_encoder(cfg, device, freeze=False):
    return DINOVforIR(cfg['num_classes'], cfg['in_features'], freeze_dino=True).to(f'cuda:{device}') 


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
    

def init_mae_skeleton_pretrained(cfg, device ,freeze=False):

    model = TransformerModel(75 ,512, depth=2)
    #print(model)
    cktp = torch.load('/home/bas06400/Thesis/checkpoint_best_transformer.pth', map_location=f'cuda:{device}')
    
    model.load_state_dict(cktp['model'])
    if freeze == True:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    return model


def init_omnivore_for_ceval(cfg, device, freeze=False):
    """
    Initialize Omnivore model specifically for cross-view evaluation (CEVAL) on DAA.

    
    Loads weights from an alignment checkpoint, adjusting for DataParallel keys and non
    matching encoder keys.
    

    Args:
        cfg (dict): Configuration containing 'in_features', 'cktp_dir', and 'aligned_model'.
        device (torch.device): The device to load the model on.
        freeze (bool): Initial freezing strategy (default: False).

    Returns:
        nn.Module: Initialized and customized Omnivore model for CEVAL.
    """
    model = omnivore_swinB_imagenet21k()
    
    model.heads = nn.Linear(1024, cfg['in_features'], bias=False)
    if freeze:
        # Freeze all parameters in the model
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze the parameters in the newly initialized head
        for param in model.heads.parameters():
            param.requires_grad = True
    cktp = torch.load(os.path.join(cfg['cktp_dir'],cfg['aligned_model']), map_location=f"cuda:{device}")
    # Adjust for DataParallel state_dict keys
    new_state_dict = {k.replace('module.modalities_encoders.ir.module.', ''): v for k, v in cktp['model_state_dict'].items()}
    model.load_state_dict(new_state_dict, strict=False)  
    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.heads.parameters():
            param.requires_grad = True  
    return model.to(f'cuda:{device}')