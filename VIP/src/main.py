import argparse
import json
import logging

import sys 
import os
import getpass

import re
import subprocess
import torch

from zeta.data_loader import load_dataloaders
from zeta.model_init import initialize_vip_encoder, MultiModalityModel
from zeta.align_process import align_modalities_process

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
def setup_logging(log_file='project.log'):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
# used modalities
# epochs 
#num workers
# dataset





# task 1 algin modalities 

def align_modalities(modalities, train_loader, val_loader, num_epochs, learning_rate, temperature, resume_from_checkpoint, checkpoint_dir, config):
    logging.info("Aligning modalities...")
    
    selected_gpu_ids = select_gpus(num_gpus=4)
    
    

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

    
    
    #torch.cuda.set_device(sorted(selected_gpu_ids)[0])
    
    multi_modality_model = MultiModalityModel(modalities_encoders, config['num_classes'], config['in_features']).cuda(sorted(selected_gpu_ids)[0])

    
    
    multi_modality_model = torch.nn.DataParallel(multi_modality_model, device_ids=sorted(selected_gpu_ids))
    

    

    print('device ids',multi_modality_model.device_ids)
    
    align_modalities_process(multi_modality_model,
                            train_loader,
                            val_loader, 
                            num_epochs, 
                            learning_rate, 
                            temperature,
                            resume_from_checkpoint,
                            checkpoint_dir,
                            config
                            )


# task 2 train classefiers for algiened encoders
def train_classifiers():
    logging.info("Training classifiers...")
    # Your code for task 2
# task 2 b continue training for classefiers

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
    setup_logging()

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
        train_classifiers()
    elif task == '3':
        evaluate_text_encoder()
    elif task == '4':
        evaluate_knn()

if __name__ == "__main__":
    main()