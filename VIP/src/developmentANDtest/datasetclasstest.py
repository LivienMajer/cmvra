from VIP.src.datasets.dataset_video_retrieval import HDVILAVideoRetrievalDataset
cfg = "/home/bas06400/Thesis/cfg.json"
import json

class Config:
    def __init__(self, json_config):
        self.train_datasets = json_config['train_datasets']
        self.val_datasets = json_config['val_datasets']
        self.inference_datasets = json_config['inference_datasets']
        self.train_n_clips = json_config['train_n_clips']
        self.train_num_frms = json_config['train_num_frms']
        self.test_n_clips = json_config['test_n_clips']
        self.test_num_frms = json_config['test_num_frms']
        self.sample_rate = json_config['sample_rate']
        self.sample_jitter = json_config['sample_jitter']
        self.video_res = json_config['video_res']
        self.input_res = json_config['input_res']
        self.max_txt_len = json_config['max_txt_len']
        self.e2e_weights_path = json_config['e2e_weights_path']
        self.clip_weights = json_config['clip_weights']
        self.clip_config = json_config['clip_config']
        self.clip_vision_additional_config = json_config['clip_vision_additional_config']
        self.train_batch_size = json_config['train_batch_size']
        self.test_batch_size = json_config['test_batch_size']
        self.max_n_example_per_group = json_config['max_n_example_per_group']
        self.gradient_accumulation_steps = json_config['gradient_accumulation_steps']
        self.n_workers = json_config['n_workers']
        self.pin_mem = json_config['pin_mem']
        self.fp16 = json_config['fp16']
        self.amp_level = json_config['amp_level']
        self.seed = json_config['seed']
        self.optim = json_config['optim']
        self.betas = json_config['betas']
        self.learning_rate = json_config['learning_rate']
        self.weight_decay = json_config['weight_decay']
        self.lr_mul = json_config['lr_mul']
        self.lr_mul_prefix = json_config['lr_mul_prefix']
        self.loss_config = json_config['loss_config']
        self.warmup_ratio = json_config['warmup_ratio']
        self.decay = json_config['decay']
        self.grad_norm = json_config['grad_norm']
        self.num_train_epochs = json_config['num_train_epochs']
        self.min_valid_steps = json_config['min_valid_steps']
        self.num_valid = json_config['num_valid']
        self.only_valid_steps = json_config['only_valid_steps']
        self.save_steps_ratio = json_config['save_steps_ratio']
        self.output_dir = json_config['output_dir']
        self.if_tb_log = json_config['if_tb_log']
        self.if_model_saver = json_config['if_model_saver']
        self.if_log2file = json_config['if_log2file']
        self.dummy_data = json_config['dummy_data']

# Load the JSON configuration
cfg_path = "/home/bas06400/Thesis/cfg.json"
with open(cfg_path, 'r') as f:
    json_config = json.load(f)

# Create a Config object
cfg = Config(json_config)

vis_dir = '/home/bas06400/ntu/nturgb+d_rgb'
anno_path = '/home/bas06400/ntu/annotations_train.jsonl'

dataset = HDVILAVideoRetrievalDataset(cfg, vis_dir, anno_path, vis_format='video', mode="train")
print(dataset[0]['video'].shape)