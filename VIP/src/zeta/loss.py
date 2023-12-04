import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss1(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss1, self).__init__()
        self.temperature = temperature

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
                sim_matrix = torch.matmul(feature_sets[i], feature_sets[j].T) / self.temperature

                # Positive pairs are the diagonal elements
                positives = torch.diag(sim_matrix)

                # Loss for this pair of modalities
                loss = -torch.log(torch.exp(positives) / torch.exp(sim_matrix).sum(dim=1))
                total_loss += loss.mean()
                count += 1

        # Average loss over all modality pairs
        total_loss /= count
        return total_loss
    

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
