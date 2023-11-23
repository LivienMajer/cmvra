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

from zeta.data_loader import load_dataloaders
from zeta.model_init import initialize_vip_encoder, MultiModalityModel
from zeta.align_process import align_modalities_process
from zeta.train_classefier import train_classefier_process

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
    log_file = f'task_{config["task"]}_{modalities_str}_{current_time}.log'
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
    logging.info("Aligning modalities...")
    
    selected_gpu_ids = select_gpus(num_gpus=int(config['number_gpus']))
    logging.info(f"Training on the following GPUs {selected_gpu_ids}")
    

    modalities_encoders = {}
    for modality in modalities:
        if modality == 'rgb':
            freeze = True
        else:
            freeze = False
        encoder = initialize_vip_encoder(config, modality=modality, freeze=freeze)
        # for some reason my encoders have to be a Dataparallel object to otherwise they dodge the wrapping of the parent model
        encoder = encoder.cuda(sorted(selected_gpu_ids)[0])
        modalities_encoders[modality] = torch.nn.DataParallel(encoder, device_ids=sorted(selected_gpu_ids))

    
    multi_modality_model = MultiModalityModel(modalities_encoders, config['num_classes'], config['in_features']).cuda(sorted(selected_gpu_ids)[0])

    multi_modality_model = torch.nn.DataParallel(multi_modality_model, device_ids=sorted(selected_gpu_ids))
    
    
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
    logging.info("Training classifiers...")

    selected_gpu_ids = select_gpus(num_gpus=int(config['number_gpus']))
    logging.info(f"Training on the following GPUs {selected_gpu_ids}")
    
    

    modalities_encoders = {}
    for modality in config['modalities']:
        
        freeze = True
        encoder = initialize_vip_encoder(config, modality=modality, freeze=freeze)
        # for some reason my encoders have to be a Dataparallel object to otherwise they dodge the wrapping of the parent model
        encoder = encoder.cuda(sorted(selected_gpu_ids)[0])
        modalities_encoders[modality] = torch.nn.DataParallel(encoder, device_ids=sorted(selected_gpu_ids))

    
    
    multi_modality_model = MultiModalityModel(modalities_encoders, config['num_classes'], config['in_features']).cuda(sorted(selected_gpu_ids)[0])

    
    
    multi_modality_model = torch.nn.DataParallel(multi_modality_model, device_ids=sorted(selected_gpu_ids))

    target_device = f'cuda:{sorted(selected_gpu_ids)[0]}'
    cktp = torch.load(os.path.join(config['cktp_dir'],config['aligned_model']), map_location=target_device)


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
    
    
    
    multi_modality_model.load_state_dict(cktp['model_state_dict'])

    #multi_modality_model = multi_modality_model.cuda(sorted(selected_gpu_ids)[0])
    train_classefier_process(multi_modality_model, 
                            sorted(selected_gpu_ids)[0], 
                            train_loader, 
                            val_loader, 
                            test_loader, 
                            config)

def evaluate_knn():
    logging.info("Evaluating KNN...")
    # Your code for task 4

# task 3 evaluate on text encoder

def evaluate_text_encoder():
    logging.info("Evaluating text encoder...")
    # Your code for task 3

# task 4 evealuate knn

def main():
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
    data_list= config['data_list']
    data_root= config['data_root']
    batch_size= config['batch_size']
    pin_memory= config['pin_memory']

    train_data, val_data, test_data = load_dataloaders(data_list=data_list,
                                                        data_root=data_root,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory)
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
        evaluate_text_encoder()
    elif task == '4':
        evaluate_knn()

if __name__ == "__main__":
    main()