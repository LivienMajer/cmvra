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
from zeta.model_init import initialize_vip_encoder, MultiModalityModel, initialize_vip_text_encoder, init_mae_model, init_mae_encoder
from zeta.align_process import align_modalities_process, eval_loss_process
from zeta.train_classefier import train_classefier_process, eval_rgb_classefier_on_ir, train_mae_classifier
from zeta.eval_vip_textencoder import eval_text_encoder_process
from zeta.mae_encoder_training import mae_training

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
    logging.info("Aligning modalities......")
    
    selected_gpu_ids = select_gpus(num_gpus=int(config['number_gpus']))
    logging.info(f"Training on the following GPUs {selected_gpu_ids}")
    

    modalities_encoders = {}
    for modality in modalities:
        if modality == 'rgb':
            freeze = True
        else:
            freeze = False
        encoder = initialize_vip_encoder(config, modality=modality, freeze=freeze)
        # for some reason my encoders have to be a Dataparallel object too otherwise they dodge the wrapping of the parent model
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
        if config['full_train_classifiers']:
            logging.info("Encoders are now trainable")
            freeze = False
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
    
    
    if not config['full_train_classifiers']:
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

    #multi_modality_model = multi_modality_model.cuda(sorted(selected_gpu_ids)[0])
    train_classefier_process(multi_modality_model, 
                            sorted(selected_gpu_ids)[0], 
                            train_loader, 
                            val_loader, 
                            test_loader, 
                            config)

# task 3 evaluate loss for unchanged VIP encoders. Expand ir and depth dims
def eval_loss(modalities, train_loader, val_loader, num_epochs, learning_rate, temperature, resume_from_checkpoint, checkpoint_dir, config):
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
    
    logging.info("Evaluing alginment with VIP text encoder...")

    selected_gpu_ids = select_gpus(num_gpus=int(config['number_gpus']))
    logging.info(f"Evaluing on the following GPUs {selected_gpu_ids}")
    
    
    

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
    text_model = initialize_vip_text_encoder(config, target_device)
    text_model = text_model.to(target_device)
    cktp = torch.load(os.path.join(config['cktp_dir'],config['aligned_model']), map_location=target_device)
    #print(cktp['model_state_dict'].keys())
    multi_modality_model.load_state_dict(cktp['model_state_dict'])

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
    logging.info(f"Training MAE for {config['modalities'][0]}...")
    # Your code for task 4
    selected_gpu_ids = select_gpus(num_gpus=int(config['number_gpus']))
    logging.info(f"Evaluing on the following GPUs {selected_gpu_ids}")
    if len(config['modalities']) > 1:
        logging.warn('Only single modalitiy Mae training implemented')

    
    if config['train_classifier'] == True:
        device = sorted(selected_gpu_ids)[0] 
        encoder , classifier = init_mae_encoder(config)
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

if __name__ == "__main__":
    
    main()