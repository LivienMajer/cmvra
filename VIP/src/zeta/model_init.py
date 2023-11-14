import json
import torch
from easydict import EasyDict as edict
from modeling.CLIP_ViP import CLIPVisionModel, CLIPVisionTransformer
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPVisionConfig
from transformers import CLIPPreTrainedModel
from torch import nn
from modeling.VidCLIP import VidCLIP

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def initialize_vip_encoder(config_path, weights_path, modality='rgb', freeze=True):

    # Load configuration
    with open(config_path) as f:
        config = json.load(f)

    # Create an 'args' object from the configuration
    args = edict({
        "clip_config": config["clip_config"],
        "clip_weights": config["clip_weights"],
        "clip_vision_additional_config": edict(config["clip_vision_additional_config"]),
        "e2e_weights_path": config["e2e_weights_path"]
    })

    # Initialize the model instance
    model_instance = VidCLIP(args)
    print(model_instance)

    # Load model weights
    ckpt = torch.load(weights_path)
    model_instance.load_state_dict(ckpt)

    # Load the base CLIPConfig
    clipconfig = CLIPVisionConfig.from_pretrained(args.clip_config)

    additional_vision_config_obj = SimpleNamespace(**config["additional_vision_config"])
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

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

    if freeze:
        # Freeze the parameters if required
        for param in model.parameters():
            param.requires_grad_(False)

    # adjust channel dim in patch projection layer
    if modality in ['ir', 'depth', 'skeleton']:
        model.vision_model.embeddings.patch_embedding = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16), bias=False)

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
        "e2e_weights_path": config["e2e_weights_path"]
    })

    # Load the base CLIPConfig
    clipconfig = CLIPVisionConfig.from_pretrained(args.clip_config)

    additional_vision_config_obj = SimpleNamespace(**config["additional_vision_config"])
    setattr(clipconfig, "additional_vision_config", additional_vision_config_obj)

    # Initialize the CLIPVisionModel
    model = CLIPVisionModel(clipconfig)



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
    def __init__(self, modalities_encoders, num_classes, in_features):
        super(MultiModalityModel, self).__init__()
        self.modalities_encoders = nn.ModuleDict(modalities_encoders)

        # Adding a linear layer for each modality
        for modality in modalities_encoders:
            setattr(self, f"{modality}_classifier", nn.Linear(in_features, num_classes))

    def forward_encoder(self, modality, x):
        if modality in self.modalities_encoders:
            return self.modalities_encoders[modality](x)
        else:
            raise ValueError(f"Modality {modality} not recognized")

    def forward_classifier(self, modality, x):
        encoder_output = self.forward_encoder(modality, x)
        classifier = getattr(self, f"{modality}_classifier")
        return classifier(encoder_output)