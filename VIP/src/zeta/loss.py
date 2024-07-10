import torch
import torch.nn as nn
import torch.nn.functional as F

class NCEContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, temp):
        super(NCEContrastiveLoss, self).__init__()
        self.temp = temp

    def forward(self, vis_feat, text_feat):

        vis_feat, text_feat = normalize(vis_feat, text_feat)
        t2v = torch.matmul(vis_feat, text_feat.permute(1, 0)) / self.temp  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        loss =    (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label) ) / 2
        return loss

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class DiagonalMSEContrastiveLoss(nn.Module):
    """
    Compute Mean Squared Error (MSE) loss between corresponding embeddings in the batch
    across two modalities (i.e., diagonal pairs only).
    """

    def __init__(self):
        super(DiagonalMSEContrastiveLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # Default reduction is 'mean'

    def forward(self, vis_feat, text_feat):
        # Ensure input features have the same dimensions
        if vis_feat.size(0) != text_feat.size(0):
            raise ValueError("The number of features in each set must match")

        # Compute MSE loss directly between corresponding elements
        loss = self.mse_loss(vis_feat, text_feat)
        return loss


class SoftLabelCrossEntropyLoss(nn.Module):
    """
    Compute cross-entropy loss using label smoothing for soft labeling.
    This involves setting the label_smoothing parameter of the CrossEntropyLoss.
    """
    def __init__(self, num_classes=34, smoothing=0.1):
        super(SoftLabelCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.smoothing)

    def forward(self, logits, hard_labels):
        # hard_labels are expected to be class indices; for true soft targets, you would need a different approach
        return self.loss_fn(logits, hard_labels)
    

class DiagonalKLDivLoss(nn.Module):
    """
    Compute KL divergence loss between logits and soft targets.
    This uses log_softmax on logits and assumes targets are given as
    probability distributions.
    """
    def __init__(self, temperature=1.0):
        super(DiagonalKLDivLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, targets):
        if logits.size(0) != targets.size(0):
            raise ValueError("The number of features in each set must match")

        # Apply softmax to targets to convert them into probability distributions
        targets = F.softmax(targets / self.temperature, dim=-1)

        # Apply log_softmax to logits
        logits = F.log_softmax(logits / self.temperature, dim=-1)

        # Compute KL divergence
        loss = self.kl_div(logits, targets)
        return loss
    
class InfoNCELoss1(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss1, self).__init__()
        self.temperature = temperature
        # Instantiate NCEContrastiveLoss with the given temperature
        self.nce_loss = NCEContrastiveLoss(temperature)

    def forward(self, *feature_sets):
        num_modalities = len(feature_sets)
        total_loss = 0.0
        count = 0
        loss_dict = {}

        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Calculate loss for each pair using NCEContrastiveLoss
                # Here, we consider only one direction (i -> j)
                loss_ij = self.nce_loss(feature_sets[i], feature_sets[j])
                total_loss += loss_ij
                count += 1
                # Detach the loss, move it to CPU, and convert to Python scalar
                loss_value = loss_ij.detach().cpu().item()
                loss_dict[f'modality_{i}_to_modality_{j}'] = loss_value

        # Average loss over all modality pairs
        total_loss /= count
        return total_loss, loss_dict
    

class SigmoidContrastiveMultiModalLoss(nn.Module):
    def __init__(self, temperature_initial=10, bias_initial=-10):
        super(SigmoidContrastiveMultiModalLoss, self).__init__()
        # Initialize temperature and bias as learnable parameters
        self.temperature = nn.Parameter(torch.tensor([temperature_initial]).float())
        self.bias = nn.Parameter(torch.tensor([bias_initial]).float())

    def forward(self, *feature_sets):
        """
        *feature_sets are the normalized feature vectors from the different modalities.
        Each element in feature_sets should have the shape [batch_size, feature_size].
        """
        num_modalities = len(feature_sets)
        batch_size = feature_sets[0].size(0)

        total_loss = 0.0
        count = 0

        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Compute similarity scores between features of modality i and j
                logits = torch.matmul(feature_sets[i], feature_sets[j].T)
                logits = logits * self.temperature.exp().to(logits.device) + self.bias.to(logits.device)

                # Create labels: 1 for matching pairs (diagonal), -1 for non-matching pairs
                labels = 2 * torch.eye(batch_size).to(logits.device) - 1

                # Compute the sigmoid loss for this pair of modalities
                loss = -torch.mean(F.logsigmoid(labels * logits))
                total_loss += loss
                count += 1

        # Average loss over all modality pairs
        total_loss /= count
        return total_loss
    
# Loss function
def mse_loss(reconstructed, original):
    criterion = nn.MSELoss()
    loss = criterion(reconstructed, original)
    return loss

def create_scheduler(optimizer, config):
    scheduler_config = config.get('scheduler_config', {})
    scheduler_type = scheduler_config.get('type', 'step')
    scheduler_params = scheduler_config.get('params', {})

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler