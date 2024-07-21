"""
Main script for multi-modal learning tasks.

This script serves as the entry point for various multi-modal learning tasks including:
1. Aligning modalities
2. Training classifiers
3. Evaluating loss for unchanged VIP encoders
4. Evaluating alignment with VIP text encoder
5. Training MAE Encoder
6. Performing KNN evaluation

The script uses a configuration file to set up the environment and parameters for each task.

Usage:
    python main.py --config path/to/config.json

The config file should be a JSON file containing all necessary parameters for the selected task.
"""



import argparse
import json
import logging
import datetime

import sys 
import os
import getpass

import re
import subprocess
import torch

import multiprocessing
import time

from zeta.data_loader import load_dataloaders
from zeta.model_init import initialize_vip_encoder, MultiModalityModel, initialize_vip_text_encoder, init_mae_model, init_mae_encoder, init_omnivore_encoder, init_dino_encoder, init_mae_skeleton_pretrained, init_omnivore_for_ceval
from zeta.align_process import align_modalities_process, eval_loss_process
from zeta.train_classefier import train_classefier_process, eval_rgb_classefier_on_ir, train_mae_classifier, MultiModalityClassifierTrainer
from zeta.eval_vip_textencoder import eval_text_encoder_process
from zeta.mae_encoder_training import mae_training
from zeta.eval_knn import eval_knn

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




def worker(num):
    """thread worker function"""
    print(f'Worker: {num}')
    time.sleep(2)

def get_gpu_memory_map():
    """Returns a dictionary of GPU ID to memory available in MB"""
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'])
    gpu_info = [x.split(', ') for x in result.decode('utf-8').strip().split('\n')]
    return {int(info[0]): int(info[1]) for info in gpu_info}

def select_gpus(num_gpus=2):
    gpu_memory_map = get_gpu_memory_map()
    selected_gpus = sorted(gpu_memory_map, key=gpu_memory_map.get, reverse=True)[:num_gpus]
    return selected_gpus

def parse_args():
    parser = argparse.ArgumentParser(description="Project Description")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    return config

# cmd arguments
def setup_logging(config):
    modalities_str = '_'.join(config['modalities'])
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'task_{config["task"]}_{modalities_str}_{config["encoder_model"]}_{config["dataset"]}_{config["split"]}_{current_time}_{config["topic"]}.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    # Log the entire config
    config_str = json.dumps(config, indent=4)
    logging.info(f"Configuration:\n{config_str}")


# task 1 algin modalities 

def align_modalities(modalities, train_loader, val_loader, num_epochs, learning_rate, temperature, resume_from_checkpoint, checkpoint_dir, config):
    """
    Align different modalities using a multi-modality model.

    This function initializes encoders for each modality, creates a multi-modality model,
    and trains it to align the representations of different modalities.

    Args:
        modalities (list): List of modalities to be aligned.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        temperature (float): Temperature parameter for the loss function.
        resume_from_checkpoint (bool): Whether to resume training from a checkpoint.
        checkpoint_dir (str): Directory to save checkpoints.
        config (dict): Configuration dictionary containing model and training settings.

    Returns:
        None
    """
    logging.info("Aligning modalities......")
    
    selected_gpu_ids = select_gpus(num_gpus=int(config['number_gpus']))
    logging.info(f"Training on the following GPUs {selected_gpu_ids}")
    
    device = sorted(selected_gpu_ids)[0]
    modalities_encoders = {}
    
    for i, modality in enumerate(modalities):
        freeze = modality == 'rgb' and config['bind_to_rgb']
        
        # Determine the encoder model for the current modality
        if config['encoder_model'] == 'MIX':
            assert modality in config['modalities_encoders'], f"Encoder type for modality '{modality}' is not specified in 'modalities_encoders' config."
            encoder_model = config['modalities_encoders'][modality]
            #print(encoder_model)
        else:
            encoder_model = config['encoder_model']
        
        if encoder_model == 'CLIP-VIP':
            encoder = initialize_vip_encoder(config, modality=modality, freeze=freeze)
        elif encoder_model == 'MAE':
            assert len(modalities) == len(config.get('trained_encoder', [])), f"Length of modalities list ({len(modalities)}) and length of path list for trained mae encoders ({len(config.get('trained_encoder', []))}) do not match"
            checkpoint_name = config['trained_encoder'][i]  # This gets the checkpoint name for the current modality
            assert modality in checkpoint_name, f"The checkpoint {checkpoint_name} does not match the modality {modality}."
            encoder = init_mae_encoder(config, checkpoint_name, device, return_class=False, freeze=freeze)
        elif encoder_model == 'OMNIVORE':
            encoder = init_omnivore_encoder(config, device, freeze=freeze)
        elif encoder_model == 'DINO':
            encoder = init_dino_encoder(config, device ,freeze=freeze)
        elif encoder_model == 'MAEPS':
            encoder = init_mae_skeleton_pretrained(config, device ,freeze=freeze)
        else:
            logging.warn(f"Unsupported encoder model option: {encoder_model} for modality {modality}")
        # for some reason my encoders have to be a Dataparallel object too otherwise they dodge the wrapping of the parent model
        modalities_encoders[modality] = torch.nn.DataParallel(encoder, device_ids=sorted(selected_gpu_ids))
    
    multi_modality_model = MultiModalityModel(modalities_encoders, config['num_classes'], config['in_features']).cuda(sorted(selected_gpu_ids)[0])

    multi_modality_model = torch.nn.DataParallel(multi_modality_model, device_ids=sorted(selected_gpu_ids))
    #print(multi_modality_model)

    if config['align_pre_training'] == True:
        target_device = f'cuda:{sorted(selected_gpu_ids)[0]}'
        cktp = torch.load(os.path.join(config['cktp_dir'],config['aligned_model']), map_location=target_device)

        #if not config['full_train_classifiers']:
        # Prepare a new state_dict for the model
        new_state_dict = {}

        for name, param in cktp['model_state_dict'].items():
            # If the shape of the pretrained parameter doesn't match the model, skip it
            if name in multi_modality_model.state_dict() and param.size() == multi_modality_model.state_dict()[name].size():
                new_state_dict[name] = param
            else:
                # Log or print the mismatch information for debugging
                
                logging.warning(f"Skipping loading parameter: {name} due to size mismatch.")

        # Load the updated state_dict
        multi_modality_model.load_state_dict(new_state_dict, strict=False)
    align_modalities_process(multi_modality_model,
                            train_loader,
                            val_loader, 
                            num_epochs, 
                            learning_rate, 
                            temperature,
                            resume_from_checkpoint,
                            checkpoint_dir,
                            sorted(selected_gpu_ids)[0],
                            config
                            )


# task 2 train classefiers for algiened encoders
def train_classifiers(train_loader, val_loader, test_loader, config):
    """
    Train classifiers for aligned encoders.

    This function initializes a multi-modality model with pre-trained encoders,
    loads aligned weights, and trains classifiers for each modality.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        config (dict): Configuration dictionary containing model and training settings.

    Returns:
        None
    """
    logging.info("Training classifiers...")

    selected_gpu_ids = select_gpus(num_gpus=int(config['number_gpus']))
    logging.info(f"Training on the following GPUs {selected_gpu_ids}")
    
    

    modalities_encoders = {}
    """
    for modality in config['modalities']:
        
        freeze = True
        if config['full_train_classifiers']:
            logging.info("Encoders are now trainable")
            freeze = False
        encoder = initialize_vip_encoder(config, modality=modality, freeze=freeze)
        # for some reason my encoders have to be a Dataparallel object to otherwise they dodge the wrapping of the parent model
        encoder = encoder.cuda(sorted(selected_gpu_ids)[0])
        """
    ########
    modalities = config['modalities']
    device = sorted(selected_gpu_ids)[0]
    
    for i, modality in enumerate(modalities):
        freeze = True
        if config['full_train_classifiers']:
            logging.info("Encoders are now trainable")
            freeze = False
        # Determine the encoder model for the current modality
        if config['encoder_model'] == 'MIX':
            assert modality in config['modalities_encoders'], f"Encoder type for modality '{modality}' is not specified in 'modalities_encoders' config."
            encoder_model = config['modalities_encoders'][modality]
        else:
            encoder_model = config['encoder_model']
        
        if encoder_model == 'CLIP-VIP':
            encoder = initialize_vip_encoder(config, modality=modality, freeze=freeze)
        elif encoder_model == 'MAE':
            assert len(modalities) == len(config.get('trained_encoder', [])), f"Length of modalities list ({len(modalities)}) and length of path list for trained mae encoders ({len(config.get('trained_encoder', []))}) do not match"
            checkpoint_name = config['trained_encoder'][i]  # This gets the checkpoint name for the current modality
            assert modality in checkpoint_name, f"The checkpoint {checkpoint_name} does not match the modality {modality}."
            encoder = init_mae_encoder(config, checkpoint_name, device, return_class=False, freeze=freeze)
        elif encoder_model == 'OMNIVORE':
            if modality in ['rgb','ir','depth','skeleton']:
                encoder = init_omnivore_encoder(config, device, freeze=freeze)
            else:
                encoder = init_omnivore_for_ceval(config, device, freeze=freeze)
        elif encoder_model == 'DINO':
            encoder = init_dino_encoder(config, device, freeze=freeze)
        elif encoder_model == 'MAEPS':
            encoder = init_mae_skeleton_pretrained(config, device ,freeze=freeze)
        else:
            logging.warn(f"Unsupported encoder model option: {encoder_model} for modality {modality}")
        ############
        modalities_encoders[modality] = torch.nn.DataParallel(encoder, device_ids=sorted(selected_gpu_ids))

    
    
    multi_modality_model = MultiModalityModel(modalities_encoders, config['num_classes'], config['in_features']).cuda(sorted(selected_gpu_ids)[0])

    
    
    multi_modality_model = torch.nn.DataParallel(multi_modality_model, device_ids=sorted(selected_gpu_ids))

    target_device = f'cuda:{sorted(selected_gpu_ids)[0]}'
    cktp = torch.load(os.path.join(config['cktp_dir'],config['aligned_model']), map_location=target_device)

    
    """ for ir cross evals
    checkpoint = torch.load(os.path.join(config['cktp_dir'],config['aligned_model']), map_location=target_device)

   
    ir_classifier_weights = {k: v for k, v in checkpoint['model_state_dict'].items() if 'ir_classifier' in k}
    classifier_names = [
    "steering_wheel_classifier",
    "a_column_driver_classifier",
    "a_column_co_driver_classifier",
    "ceiling_classifier",
    "inner_mirror_classifier"
]

    for classifier_name in classifier_names:
        # Construct new state_dict for this classifier
        new_classifier_weights = {k.replace('ir_classifier', classifier_name): v for k, v in ir_classifier_weights.items()}
        logging.info(list(new_classifier_weights.keys()))
        # Load the ir_classifier weights into the current classifier
        multi_modality_model.load_state_dict(new_classifier_weights, strict=False)
        new_classifier_weights2 = {k.replace('modalities_encoders.ir.module.heads', f"modalities_encoders.{classifier_name[:-11]}.module.heads"): v for k, v in checkpoint['model_state_dict'].items()}
        new_classifier_weights2_keys = list(new_classifier_weights2.keys())
        #logging.info(new_classifier_weights2_keys)
        #logging.info(list(multi_modality_model.state_dict().keys()))
        multi_modality_model.load_state_dict(new_classifier_weights2, strict=False)

    def compare_parameters(param1, param2):
        for p1, p2 in zip(param1, param2):
            if not torch.equal(p1, p2):
                return False
        return True

    # After loading the parameters into a classifier
    for classifier_name in classifier_names:
        # Extract parameters from both classifiers for comparison
        ir_params = [v for k, v in ir_classifier_weights.items() if 'weight' in k or 'bias' in k]
        current_params = [v for k, v in multi_modality_model.state_dict().items() if classifier_name in k and ('weight' in k or 'bias' in k)]
        
        # Compare the parameters
        are_params_equal = compare_parameters(ir_params, current_params)
        
        assert are_params_equal, f"Parameters do not match exactly for {classifier_name}"
        logging.info(f"Parameters successfully verified for {classifier_name}.")
    """

    """
    cktp_state_dict = cktp['model_state_dict']
    
    # Get your current model's state dictionary
    current_state_dict = multi_modality_model.state_dict()

    # List for storing keys that do not match
    mismatched_keys = []

    # Compare
    for key in cktp_state_dict:
        if key in current_state_dict:
            # Check if the weights are the same
            if not torch.equal(cktp_state_dict[key], current_state_dict[key]):
                mismatched_keys.append(key)
                logging.info(f"Weights differ for {key}")
        else:
            mismatched_keys.append(key)
            logging.info(f"{key} is not present in the current model's state dictionary")

    # Logging the final list of mismatched keys
    if mismatched_keys:
        logging.info("Final list of mismatched keys: " + ", ".join(mismatched_keys))
    else:
        logging.info("All keys matched successfully.")
    """
    
    
    #if not config['full_train_classifiers']:
    # Prepare a new state_dict for the model
    new_state_dict = {}

    for name, param in cktp['model_state_dict'].items():
        # If the shape of the pretrained parameter doesn't match the model, skip it
        if name in multi_modality_model.state_dict() and param.size() == multi_modality_model.state_dict()[name].size():
            new_state_dict[name] = param
        else:
            # Log or print the mismatch information for debugging
            pass
            #logging.warning(f"Skipping loading parameter: {name} due to size mismatch.")

    # Load the updated state_dict
    multi_modality_model.load_state_dict(new_state_dict, strict=False)
        
    if config['full_train_classifiers']:
        logging.info('Setting grads')
        for param in multi_modality_model.module.modalities_encoders.parameters():
            param.requires_grad_(True)

    #multi_modality_model = multi_modality_model.cuda(sorted(selected_gpu_ids)[0])
    trainer = MultiModalityClassifierTrainer(multi_modality_model, 
                            sorted(selected_gpu_ids)[0], 
                            train_loader, 
                            val_loader, 
                            test_loader, 
                            config)
    trainer.train()
    """
    train_classefier_process(multi_modality_model, 
                            sorted(selected_gpu_ids)[0], 
                            train_loader, 
                            val_loader, 
                            test_loader, 
                            config)
    """
# task 3 evaluate loss for unchanged VIP encoders. Expand ir and depth dims
def eval_loss(modalities, train_loader, val_loader, num_epochs, learning_rate, temperature, resume_from_checkpoint, checkpoint_dir, config):
    """
    Evaluate loss for unchanged VIP encoders. This was just a basic test without much utility.

    This function initializes VIP encoders for each modality, creates a multi-modality model,
    and evaluates the loss without training the encoders.

    Args:
        modalities (list): List of modalities to be evaluated.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of evaluation epochs.
        learning_rate (float): Learning rate (not used for evaluation, but kept for consistency).
        temperature (float): Temperature parameter for the loss function.
        resume_from_checkpoint (bool): Whether to resume from a checkpoint.
        checkpoint_dir (str): Directory to save checkpoints.
        config (dict): Configuration dictionary containing model and evaluation settings.

    Returns:
        None
    """
    logging.info("Evaluing Loss...")
    
    selected_gpu_ids = select_gpus(num_gpus=int(config['number_gpus']))
    logging.info(f"Training on the following GPUs {selected_gpu_ids}")
    

    modalities_encoders = {}
    for modality in modalities:
        encoder = initialize_vip_encoder(config, modality=modality, freeze=True)
        # for some reason my encoders have to be a Dataparallel object too otherwise they dodge the wrapping of the parent model
        encoder = encoder.cuda(sorted(selected_gpu_ids)[0])
        modalities_encoders[modality] = torch.nn.DataParallel(encoder, device_ids=sorted(selected_gpu_ids))

    
    multi_modality_model = MultiModalityModel(modalities_encoders, config['num_classes'], config['in_features']).cuda(sorted(selected_gpu_ids)[0])

    multi_modality_model = torch.nn.DataParallel(multi_modality_model, device_ids=sorted(selected_gpu_ids))

    eval_loss_process(multi_modality_model,
                            train_loader,
                            val_loader, 
                            num_epochs, 
                            learning_rate, 
                            temperature,
                            resume_from_checkpoint,
                            checkpoint_dir,
                            sorted(selected_gpu_ids)[0],
                            config
                            )


#task 4
def eval_text_encoder(train_data, 
                        val_data, 
                        test_data,
                        config):
    """
    Evaluate alignment with VIP text encoder.

    This function initializes encoders for each modality, creates a multi-modality model,
    loads aligned weights, and evaluates the alignment between visual and text encoders.

    Args:
        train_data (DataLoader): DataLoader for training data.
        val_data (DataLoader): DataLoader for validation data.
        test_data (DataLoader): DataLoader for test data.
        config (dict): Configuration dictionary containing model and evaluation settings.

    Returns:
        None
    """
    
    logging.info("Evaluing alginment with VIP text encoder...")

    selected_gpu_ids = select_gpus(num_gpus=int(config['number_gpus']))
    logging.info(f"Evaluing on the following GPUs {selected_gpu_ids}")
    
    
    

    modalities_encoders = {}
    modalities = config['modalities']
    device = sorted(selected_gpu_ids)[0]
    for i, modality in enumerate(modalities):
        freeze = True
        # Determine the encoder model for the current modality
        if config['encoder_model'] == 'MIX':
            assert modality in config['modalities_encoders'], f"Encoder type for modality '{modality}' is not specified in 'modalities_encoders' config."
            encoder_model = config['modalities_encoders'][modality]
        else:
            encoder_model = config['encoder_model']
        
        if encoder_model == 'CLIP-VIP':
            encoder = initialize_vip_encoder(config, modality=modality, freeze=freeze)
        elif encoder_model == 'MAE':
            assert len(modalities) == len(config.get('trained_encoder', [])), f"Length of modalities list ({len(modalities)}) and length of path list for trained mae encoders ({len(config.get('trained_encoder', []))}) do not match"
            checkpoint_name = config['trained_encoder'][i]  # This gets the checkpoint name for the current modality
            assert modality in checkpoint_name, f"The checkpoint {checkpoint_name} does not match the modality {modality}."
            encoder = init_mae_encoder(config, checkpoint_name, device, return_class=False, freeze=freeze)
        elif encoder_model == 'OMNIVORE':
            if modality in ['rgb','ir','depth','skeleton','rgb2','rgb3']:
                encoder = init_omnivore_encoder(config, device, freeze=freeze)
            else:
                encoder = init_omnivore_for_ceval(config, device, freeze=freeze)
        elif encoder_model == 'DINO':
            encoder = init_dino_encoder(config, device, freeze=freeze)
        elif encoder_model == 'MAEPS':
            encoder = init_mae_skeleton_pretrained(config, device ,freeze=freeze)
        else:
            logging.warn(f"Unsupported encoder model option: {encoder_model} for modality {modality}")
        modalities_encoders[modality] = torch.nn.DataParallel(encoder, device_ids=sorted(selected_gpu_ids))

    
    
    multi_modality_model = MultiModalityModel(modalities_encoders, config['num_classes'], config['in_features']).cuda(sorted(selected_gpu_ids)[0])
    multi_modality_model = torch.nn.DataParallel(multi_modality_model, device_ids=sorted(selected_gpu_ids))

    target_device = f'cuda:{sorted(selected_gpu_ids)[0]}'
    text_model = initialize_vip_text_encoder(config, target_device)
    text_model = text_model.to(target_device)
    cktp = torch.load(os.path.join(config['cktp_dir'],config['aligned_model']), map_location=target_device)
    new_state_dict = {}

    for name, param in cktp['model_state_dict'].items():
        # If the shape of the pretrained parameter doesn't match the model, skip it
        if name in multi_modality_model.state_dict() and param.size() == multi_modality_model.state_dict()[name].size():
            new_state_dict[name] = param
        else:
            # Log or print the mismatch information for debugging
            logging.warning(f"Skipping loading parameter: {name} due to size mismatch.")

    # Load the updated state_dict
    multi_modality_model.load_state_dict(new_state_dict, strict=False)

    eval_text_encoder_process(visual_model=multi_modality_model,
                              text_model=text_model,
                              train_data=train_data,
                              val_data=val_data,
                              test_data=test_data,
                              device=target_device,
                              config=config
                              )

# task 5 train Mae Encoder
def train_Mae_Encoder(train_data, val_data, test_data, config):
    """
    Train MAE (Masked Autoencoder) Encoder for a single modality.

    This function either trains a MAE model from scratch or evaluatess a pre-trained MAE encoder
    with a classifier, depending on the configuration.

    Args:
        train_data (DataLoader): DataLoader for training data.
        val_data (DataLoader): DataLoader for validation data.
        test_data (DataLoader): DataLoader for test data.
        config (dict): Configuration dictionary containing model and training settings.

    Returns:
        None
    """
    logging.info(f"Training MAE for {config['modalities'][0]}...")
    selected_gpu_ids = select_gpus(num_gpus=int(config['number_gpus']))
    logging.info(f"Evaluing on the following GPUs {selected_gpu_ids}")
    if len(config['modalities']) > 1:
        logging.warn('Only single modalitiy Mae training implemented')

    
    if config['train_classifier'] == True:
        device = sorted(selected_gpu_ids)[0] 
        encoder , classifier = init_mae_encoder(config, config['trained_encoder'], device)
        train_mae_classifier(encoder.to(device), 
                             classifier.to(device), 
                             train_data, 
                             val_data, 
                             test_data, 
                             device, 
                             config)
    else:
        model = init_mae_model(selected_gpu_ids, config)
        mae_training(model=model,
                 train_data=train_data,
                 val_data=val_data,
                 test_data=test_data,
                 device=sorted(selected_gpu_ids)[0],
                config=config)
        
def knn(train_data, test_data, config):
    """
    Perform k-Nearest Neighbors (kNN) evaluation on the multi-modality model.

    This function initializes encoders for each modality, creates a multi-modality model,
    loads aligned weights, and performs kNN evaluation on the test data.

    Args:
        train_data (DataLoader): DataLoader for training data.
        test_data (DataLoader): DataLoader for test data.
        config (dict): Configuration dictionary containing model and evaluation settings.

    Returns:
        None
    """
    logging.info("Evaluing knn...")

    selected_gpu_ids = select_gpus(num_gpus=int(config['number_gpus']))
    logging.info(f"Evaluing on the following GPUs {selected_gpu_ids}")
    
    
    

    modalities_encoders = {}
    modalities = config['modalities']
    device = sorted(selected_gpu_ids)[0]
    for i, modality in enumerate(modalities):
        freeze = True
        # Determine the encoder model for the current modality
        if config['encoder_model'] == 'MIX':
            assert modality in config['modalities_encoders'], f"Encoder type for modality '{modality}' is not specified in 'modalities_encoders' config."
            encoder_model = config['modalities_encoders'][modality]
        else:
            encoder_model = config['encoder_model']
        
        if encoder_model == 'CLIP-VIP':
            encoder = initialize_vip_encoder(config, modality=modality, freeze=freeze)
        elif encoder_model == 'MAE':
            assert len(modalities) == len(config.get('trained_encoder', [])), f"Length of modalities list ({len(modalities)}) and length of path list for trained mae encoders ({len(config.get('trained_encoder', []))}) do not match"
            checkpoint_name = config['trained_encoder'][i]  # This gets the checkpoint name for the current modality
            assert modality in checkpoint_name, f"The checkpoint {checkpoint_name} does not match the modality {modality}."
            encoder = init_mae_encoder(config, checkpoint_name, device, return_class=False, freeze=freeze)
        elif encoder_model == 'OMNIVORE':
            if modality in ['rgb','ir','depth','skeleton','rgb2','rgb3']:
                encoder = init_omnivore_encoder(config, device, freeze=freeze)
            else:
                encoder = init_omnivore_for_ceval(config, device, freeze=freeze)
        elif encoder_model == 'DINO':
            encoder = init_dino_encoder(config, device, freeze=freeze)
        elif encoder_model == 'MAEPS':
            encoder = init_mae_skeleton_pretrained(config, device ,freeze=freeze)
        else:
            logging.warn(f"Unsupported encoder model option: {encoder_model} for modality {modality}")
        modalities_encoders[modality] = torch.nn.DataParallel(encoder, device_ids=sorted(selected_gpu_ids))

    
    
    multi_modality_model = MultiModalityModel(modalities_encoders, config['num_classes'], config['in_features']).cuda(sorted(selected_gpu_ids)[0])
    multi_modality_model = torch.nn.DataParallel(multi_modality_model, device_ids=sorted(selected_gpu_ids))

    target_device = f'cuda:{sorted(selected_gpu_ids)[0]}'
    text_model = initialize_vip_text_encoder(config, target_device)
    text_model = text_model.to(target_device)
    cktp = torch.load(os.path.join(config['cktp_dir'],config['aligned_model']), map_location=target_device)
    new_state_dict = {}

    for name, param in cktp['model_state_dict'].items():
        # If the shape of the pretrained parameter doesn't match the model, skip it
        if name in multi_modality_model.state_dict() and param.size() == multi_modality_model.state_dict()[name].size():
            new_state_dict[name] = param
        else:
            # Log or print the mismatch information for debugging
            logging.warning(f"Skipping loading parameter: {name} due to size mismatch.")

    # Load the updated state_dict
    multi_modality_model.load_state_dict(new_state_dict, strict=False)

    eval_knn(multi_modality_model, train_data, test_data,device,config)



def main():
    setup_ccname()
    config = parse_args()
    setup_logging(config)

    #config
    task = config['task']
    modalities = config['modalities']
    epochs = config['epochs']
    res_ckpt = config['res_cktp']
    cktp_dir = config['cktp_dir']
    learning_rate = config['learning_rate']
    temperature = config['temperature']
    num_workers = config['num_workers']
    data_root= config['data_root']
    batch_size= config['batch_size']
    pin_memory= config['pin_memory']

    
    torch.set_num_threads(num_workers)

    train_data, val_data, test_data = load_dataloaders(data_root=data_root,
                                                       modalities=modalities,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory,
                                                        split=config['split'],
                                                        random_sample=config['random_sample'],
                                                        config=config)
    # Task executions
    if task == '1':
        align_modalities(modalities, 
        train_data, 
        val_data, 
        epochs, 
        learning_rate, 
        temperature,
        res_ckpt,
        cktp_dir,
        config)
    elif task == '2':
        train_classifiers(train_data, 
                        val_data, 
                        test_data, 
                        config)
    elif task == '3':
        eval_loss(modalities, 
        train_data, 
        val_data, 
        epochs, 
        learning_rate, 
        temperature,
        res_ckpt,
        cktp_dir,
        config)
    elif task == '4':
        eval_text_encoder(train_data, 
                        val_data, 
                        test_data,
                        config)
    elif task == '5':
        train_Mae_Encoder(train_data, 
                          val_data, 
                          test_data, 
                          config)
    elif task == '6':
        knn(train_data, 
            test_data, 
            config)

if __name__ == "__main__":
    
    main()