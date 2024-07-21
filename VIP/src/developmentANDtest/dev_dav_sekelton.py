import logging
from functools import partial
from omnimae import make_conv_or_linear, reshape_and_init_as_mlp
import torch
import torch.nn as nn
from omnivore import omnivore_swinB_imagenet21k
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm
from omegaconf import OmegaConf
from timm.models.layers import trunc_normal_
from torch.hub import load_state_dict_from_url
import random
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to save the RGB encoder weights
def save_best_model(model, best_val_loss, current_val_loss, epoch, save_dir='model_weights', filename='omnivore.pth'):
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filename)
        torch.save(model.state_dict(), save_path)  # Save model state_dict instead of the entire model for best practice
        logger.info(f"Saved new best RGB encoder model at epoch {epoch} to {save_path}")
    return best_val_loss

# Assuming CrossEntropyLoss is suitable for your model's output
def balanced_cross_entropy_loss(output, target, weights=None):
    if weights is not None:
        
        loss = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(output.device))
    else:
        loss = nn.CrossEntropyLoss()
    return loss(output, target)

def train(classifier, model, train_loader, optimizer_classifier, optimizer_model ,scheduler, num_epochs, val_loader, device='cuda:0', class_weights=None):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = batch_data['skeleton'].to(device), labels.to(device)
            optimizer_classifier.zero_grad()
            optimizer_model.zero_grad()

            inputs = inputs.view(inputs.size(0),inputs.size(1),-1)
            #print(f"the input shae now {inputs.shape}")
            #print(expanded_inputs.shape)
            #print(inputs.expand(-1,-1, 4, -1, -1).shape)
            src_key_padding_mask = torch.zeros((inputs.size(0),inputs.size(1)))
            outputs = classifier(model(inputs, inputs, src_key_padding_mask))
            #print(len(class_weights))
            #print(f"sis is se output shape: {outputs.shape}")
            loss = balanced_cross_entropy_loss(outputs, labels, weights=class_weights)
            
            _, predicted = torch.max(outputs.data, 1)
            total_loss += loss.item()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            loss.backward()
            optimizer_model.step()
            optimizer_classifier.step()
            #break
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        logger.info(f"Epoch: {epoch}, Average Training Loss: {avg_loss}, Training Accuracy: {accuracy}")

        if val_loader:
            model.eval()
            val_loss, val_accuracy = validate(classifier, model, val_loader, device)
            scheduler.step(val_loss)
            best_val_loss = save_best_model(model, best_val_loss, val_loss, epoch)
            logger.info(f"Epoch: {epoch}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

def validate(classifier, model, val_loader, device='cuda:0'):
    total_val_loss = 0
    total_correct = 0
    num_batches = 0

    with torch.no_grad():
        for batch_data, labels in val_loader:
            inputs, labels = batch_data['skeleton'].to(device), labels.to(device)
             
            inputs = inputs.view(inputs.size(0),inputs.size(1),-1)
            src_key_padding_mask = torch.zeros((inputs.size(0),inputs.size(1)))
            outputs = classifier(model(inputs, inputs, src_key_padding_mask))
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            num_batches += 1
            #break
    avg_val_loss = total_val_loss / num_batches
    avg_accuracy = total_correct / (num_batches * labels.size(0))
    return avg_val_loss, avg_accuracy


import torch
import torch.nn as nn
import torch.nn.functional as F



"""
Implementation of the transformer model.
Reference:
1. https://buomsoo-kim.github.io/attention/2020/04/21/Attention-mechanism-19.md/
2. http://nlp.seas.harvard.edu/annotated-transformer/#encoder-and-decoder-stacks
3. https://datascience.stackexchange.com/questions/93144/minimal-working-example-or-tutorial-showing-how-to-use-pytorchs-nn-transformerd
"""
import copy

import torch
import torch.nn as nn
from typing import Dict, Callable, Tuple, Any, Optional
from torch import Tensor
import copy


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_position_embeddings=512):
        super().__init__()
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    """
    def forward(self, x: Tensor, idx_remain=None):
        
        :param x: (B, T, C)
        :return: (B, T, C)
        
        if idx_remain is None:
            seq_length = x.size(1)
            position_ids = self.position_ids[:, :seq_length]
            x = x + self.position_embeddings(position_ids)
        else:
            x = x + self.position_embeddings(idx_remain)
        return self.dropout(x)
    """
    def forward(self, x: Tensor, idx_remain=None):
        if idx_remain is None or len(idx_remain) == 0:
            seq_length = x.size(1)
            position_ids = self.position_ids[:, :seq_length]
        else:
            # Ensure idx_remain is a tensor and handle its shape/contents appropriately
            if isinstance(idx_remain, list):  # Convert list to tensor if necessary
                idx_remain = torch.tensor(idx_remain, device=x.device, dtype=torch.long)
            position_ids = idx_remain

        if position_ids.nelement() == 0:  # Check if position_ids is still empty
            raise ValueError("position_ids is empty. Check the logic for generating position_ids.")
        
        x = x + self.position_embeddings(position_ids)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, orig_dim, hidden_dim, depth=1, num_heads=4, mlp_ratio=4, norm_first=False, activation='gelu',
                 dropout=0.1, use_cls=True, layer_norm_eps=1e-6, max_position_embeddings=512,
                 autoregressive=True, model_type='tafar', embed_mask=0.1, mae_input_size=28, mae_output_size=56):
        super().__init__()

        self.autoregressive = autoregressive
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.seq_len = mae_output_size

        self.init_model(orig_dim, hidden_dim, hidden_dim, use_cls, embed_mask, dropout, max_position_embeddings,
                        mlp_ratio, num_heads, activation, layer_norm_eps, norm_first, depth)

        """
        if model_type in ['tafar', 'bert']:
            self.init_model(orig_dim, hidden_dim, hidden_dim, use_cls, embed_mask, dropout, max_position_embeddings,
                            mlp_ratio, num_heads, activation, layer_norm_eps, norm_first, depth)
        elif 'mae' in model_type:
            self.init_model(orig_dim, mae_input_size, mae_output_size, use_cls, embed_mask, dropout,
                            max_position_embeddings, mlp_ratio, num_heads, activation, layer_norm_eps, norm_first,
                            depth)
        else:
            raise NotImplementedError
        """

        self.init_weights()  # initialization

    def init_model(self, orig_dim, enc_hidden_dim, dec_hidden_dim, use_cls, embed_mask, dropout, max_position_embeddings, mlp_ratio, num_heads,
                   activation, layer_norm_eps, norm_first, depth):
        # only one embedding layer, since both inputs for enc and dec are poses
        self.embd_layer = nn.Linear(orig_dim, enc_hidden_dim)

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_hidden_dim)) if use_cls else None

        self.embed_mask_prob = embed_mask

        # positional encoding layers
        self.enc_pe = PositionalEncoding(enc_hidden_dim, dropout, max_position_embeddings)
        self.dec_pe = PositionalEncoding(dec_hidden_dim, dropout, max_position_embeddings)

        mlp_ratio = int(mlp_ratio)
        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            enc_hidden_dim, num_heads, enc_hidden_dim * mlp_ratio, dropout,
            activation, layer_norm_eps, batch_first=True, norm_first=norm_first)

        # self.layer_norm = nn.LayerNorm(enc_hidden_dim)  was not commented out even though it was not used in forward an checkpoint does not have it

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        if self.model_type in ['tafar', 'casmae']:
            # Transformer Decoder
            dec_layer = nn.TransformerDecoderLayer(
                dec_hidden_dim, num_heads, dec_hidden_dim * mlp_ratio, dropout,
                activation, layer_norm_eps, batch_first=True, norm_first=norm_first)
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=depth)

        elif self.model_type == 'smae':
            dec_layer = nn.TransformerEncoderLayer(
                dec_hidden_dim, num_heads, dec_hidden_dim * mlp_ratio, dropout,
                activation, layer_norm_eps, batch_first=True, norm_first=norm_first)
            self.decoder = nn.TransformerEncoder(dec_layer, num_layers=depth)

        # output layer, hidden state -> pose
        self.output_layer = nn.Sequential(
            nn.Linear(dec_hidden_dim, orig_dim)
        )

        if "mae" in self.model_type:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_hidden_dim))

    def init_weights(self) -> None:
        # Based on Timm
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=.02)
            # nn.init.trunc_normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        """Based on Huggingface Bert but use truncated normal instead of normal.
        (Timm used trunc_normal in VisionTransformer)
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=.02)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    @staticmethod
    def select_memory(memory: Tensor, memory_type: str, src_key_padding_mask: Tensor,
                      ) -> Tuple[Tensor, Tensor]:
        assert memory_type in ['cls_only', 'seq_only', 'cls_with_seq']
        if memory_type == 'cls_only':
            memory = memory[:, :1, :]
            memory_key_padding_mask = src_key_padding_mask[:, :1]
        elif memory_type == 'seq_only':
            memory = memory[:, 1:, :]
            memory_key_padding_mask = src_key_padding_mask
        else:
            memory = memory
            memory_key_padding_mask = torch.cat([src_key_padding_mask[:, :1],
                                                 src_key_padding_mask], dim=1)
        return memory, memory_key_padding_mask

    @staticmethod
    def generate_subsequent_mask(sz: int, sz1: Optional[int] = None) -> Tensor:
        """Generate a causal mask (not necessarily square) for the sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        sz1 = sz1 or sz
        return torch.triu(torch.full((sz, sz1), float('-inf')), diagonal=1)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, src_embd: Tensor,
                        idx_remain=None) -> Tensor:
        # prepend the cls token for source if needed
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(src_embd.size(0), -1, -1)
            src_embd = torch.cat((cls_token, src_embd), dim=1)
            

        mask, ids_restore = None, None
        if self.model_type in ['smae']:
            # masking: length -> length * mask_ratio
            src_embd, mask, ids_restore = self.random_masking(src_embd[:, 1:, :], 0.5)

            # add positional embeddings and encoder forwarding
            src_input = self.enc_pe(src_embd, idx_remain)
            # (B, T+1, C) with cls token
            memory = self.encoder(src_input)
        else:
            # add positional embeddings and encoder forwarding
            src_input = self.enc_pe(src_embd, idx_remain)
            # (B, T+1, C) with cls token
            memory = self.encoder(src_input)
        return memory, mask, ids_restore

    def forward_decoder(self, tgt_embd: Tensor, memory: Tensor,
                        tgt_key_padding_mask: Optional[Tensor] = None,
                        memory_key_padding_mask: Optional[Tensor] = None,
                        autoregressive: Optional[bool] = False,
                        ids_restore: Optional[Tensor] = None) -> Tensor:
        tgt_input = self.dec_pe(tgt_embd)
        if ids_restore is not None:
            mask_tokens = self.mask_token.repeat(tgt_input.shape[0], ids_restore.shape[1] + 1 - tgt_input.shape[1], 1)
            x_ = torch.cat([tgt_input[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, tgt_input.shape[2]))  # unshuffle
            tgt_input = torch.cat([tgt_input[:, :1, :], x_[:, 1:, :]], dim=1)  # append cls token

        tgt_mask = None
        # generate causal mask for self-attention in decoder
        if autoregressive:
            tgt_seq_len = tgt_input.size(1)
            tgt_mask = self.generate_subsequent_mask(tgt_seq_len).to(tgt_embd)
        if self.model_type == 'tafar':
            # decoder forwarding
            dec_out = self.decoder(
                tgt_input, memory, tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        elif self.model_type == 'casmae':
            # x0, x1 = tgt_embd.shape[0], tgt_embd.shape[1]
            # embed_mask = torch.ones((x0, x1))
            # embed_mask[:int(x1 * self.embed_mask_prob)] = 1.
            # idx_parent = torch.randperm(x0)
            # idx_child = torch.randperm(x1)
            # embed_mask = embed_mask[:, idx_child]
            # embed_mask = embed_mask[idx_parent]
            # embed_mask = embed_mask.to(tgt_embd.device)
            # decoder forwarding
            memory_key_padding_mask = None
            tgt_key_padding_mask = None
            dec_out = self.decoder(
                tgt_input, memory, tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            dec_out = dec_out[:, 1:, :]
        elif self.model_type == 'smae':
            # decoder forwarding
            dec_out = self.decoder(
                tgt_input, src_key_padding_mask=None,
            )
        return dec_out

    def forward(self, src: Tensor, tgt: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_type: str = 'seq_only',
                idx_remain: list = []) -> Tuple[Tensor, Tensor]:
        """ For training

        Args:
            src: a sequence of feature vector (B, T, C)
            tgt: a sequence of feature vector (B, T', C), C = self.hidden_dim for non-autoregressive
            src_key_padding_mask: src padding mask (B, T): a ``True`` value indicates that the corresponding ``key`` value will be ignored
            tgt_key_padding_mask: tgt padding mask (B, T'): a ``True`` value indicates that the corresponding ``key`` value will be ignored
            memory_type: one of ['cls_only', 'seq_only', 'cls_with_seq']

        Returns
            a sequence of feature vector reconstructuring the original one
        """
        assert memory_type in ['cls_only', 'seq_only', 'cls_with_seq'], f'Invalid memory type: {memory_type}'
        # embed the inputs, orig dim -> hidden dim
        if self.model_type in ['bert', 'smae']:
            src = tgt
        #print(f'this is the shape {src.shape}')

        src_embd = self.embd_layer(src)
        """
        print(f"src_embd device: {src_embd.device}")
        
        # If src_key_padding_mask is a tensor, print its device. Otherwise, indicate it's not a tensor.
        if isinstance(src_key_padding_mask, torch.Tensor):
            print(f"src_key_padding_mask device: {src_key_padding_mask.device}")
        else:
            print("src_key_padding_mask is not a tensor or not provided.")
        
        # If src_key_padding_mask is a tensor, print its device. Otherwise, indicate it's not a tensor.
        if isinstance(src_key_padding_mask, torch.Tensor):
            print(f"src_key_padding_mask device: {src_key_padding_mask.device}")
        else:
            print("src_key_padding_mask is not a tensor or not provided.")
        # Since idx_remain could be a list or a tensor, check its type before trying to print its device.
        if isinstance(idx_remain, torch.Tensor):
            print(f"idx_remain device: {idx_remain.device}")
        elif isinstance(idx_remain, list):
            print("idx_remain is a list, not a tensor.")
        else:
            print("idx_remain is not provided or not a recognized type.")
        """
        
        memory, mask, ids_restore = self.forward_encoder(
            src_embd, idx_remain)

        # memory = self.layer_norm(memory)

        if self.model_type == 'bert':
            final_out = self.output_layer(memory[:, 1:, :])
            return memory, final_out
        elif 'mae' in self.model_type:
            # memory_ = torch.zeros((memory.shape[0], self.seq_len + 1, memory.shape[2])).to(memory)
            # for b in range(len(memory_)):
            #     memory_[b, idx_remain[b]] = memory[b]
            # memory = memory_
            memory_ = self.mask_token.repeat(memory.shape[0], self.seq_len + 1, 1)
            memory_[torch.arange(memory.shape[0])[:, None], idx_remain] = memory
            memory = memory_
        selected_memory, memory_key_padding_mask = self.select_memory(
            memory, memory_type, src_key_padding_mask)

        if self.model_type == 'tafar':
            # assert torch.equal(tgt, torch.zeros(B, T, self.hidden_dim).to(src.device)), \
            #     'Decoder input should be zeros for non-autoregressive decoding'
            tgt_embd = self.embd_layer(tgt)
            #     tgt_embd = tgt
        elif 'mae' in self.model_type:
            tgt_embd = memory
        #dec_out = self.forward_decoder(
        #    tgt_embd, selected_memory, tgt_key_padding_mask,
         #   memory_key_padding_mask, self.autoregressive, ids_restore)

        #final_out = self.output_layer(dec_out)
        return memory[:, 0, :] # just return cls token for downstream task

    def generate(self, src: Tensor, tgt: Tensor,
                 src_key_padding_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_type: str = 'seq_only') -> Tuple[Tensor, Tensor]:
        """For Testing"""
        assert memory_type in ['cls_only', 'seq_only', 'cls_with_seq'], f'Invalid memory type: {memory_type}'

        max_len = src.size(1)
        if self.model_type in ['bert', 'smae', 'casmae']:
            src = tgt
        src_embd = self.embd_layer(src)
        memory = self.forward_encoder(src_embd, src_key_padding_mask)
        selected_memory, memory_key_padding_mask = self.select_memory(
            memory, memory_type, src_key_padding_mask)

        if self.model_type == 'bert':
            final_out = self.output_layer(memory)
            return memory, final_out
        if self.autoregressive:
            for i in range(max_len - 1):
                tgt_embd = self.embd_layer(tgt)
                dec_out = self.forward_decoder(
                    tgt_embd, selected_memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    autoregressive=False)
                out = self.output_layer(dec_out)
                tgt = torch.cat([tgt, out[:, -1:]], dim=1)
        else:
            if self.model_type == 'tafar':
                tgt_embd = tgt
            elif self.model_type == 'smae' or self.model_type == 'casmae':
                tgt_embd = memory
            dec_out = self.forward_decoder(
                tgt_embd, selected_memory,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                autoregressive=False)
            tgt = self.output_layer(dec_out)

        return memory, tgt

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, clas_dim, dropout=0.3):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, clas_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)





if __name__ == "__main__":
    model = TransformerModel(75 ,512, depth=2)
    #print(model)
    cktp = torch.load('/home/bas06400/Thesis/checkpoint_best_transformer.pth', map_location='cuda:0')
    
    model.load_state_dict(cktp['model'])
    model.to('cuda:0')
    for param in model.parameters():
        param.requires_grad = True
    classifier = LinearClassifier(512,120 ).to('cuda:0')
    #my_model = nn.DataParallel(model, device_ids= [1, 2])
    import sys
    sys.path.append("/home/bas06400/Thesis/VIP/src/")

    from zeta.data_loader import load_dataloaders

    data_root = "/net/polaris/storage/deeplearning/ntu"
    modalities = ["skeleton"]
    batch_size = 8
    num_workers = 5
    pin_memory = True
    split = "CV"
    random_sample = False
    config = {'dataset':'NTU','encoder_model':'CLIP-VIP','mixed_frames':False}
    train_data, val_data, test_data = load_dataloaders(data_root=data_root,
                                                        modalities=modalities,
                                                            batch_size=batch_size,
                                                            num_workers=num_workers,
                                                            pin_memory=pin_memory,
                                                            split=split,
                                                            random_sample=random_sample,
                                                            config=config)
    # Define optimizer and scheduler
    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=0.001)

    
    optimizer_model = torch.optim.Adam(model.parameters(), lr=0.0001)  

    # Scheduler for the classifier's optimizer (you might need a separate one for the model's optimizer too)
    scheduler = ReduceLROnPlateau(optimizer_classifier, mode='min', factor=0.1, patience=5)

    # Number of training epochs
    num_epochs = 100
    # Assuming train_data is your DataLoader for training
    
    
    
    # Start training
    # Start training with class weights
    train(classifier, model, train_data, optimizer_classifier, optimizer_model , scheduler, num_epochs, val_data)