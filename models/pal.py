import torch
import torch.nn as nn

class PrivilegedAttributionLoss(nn.Module):
    def __init__(self):
        super(PrivilegedAttributionLoss, self).__init__()

    def forward(self, attribution_maps, prior_maps):
        # Add a small value to standard deviation to avoid division by zero
        epsilon = 1e-8

        # Calculate mean and standard deviation for each sample in the batch
        mean_al = torch.mean(attribution_maps, dim=[1, 2, 3], keepdim=True)  # Assuming BCHW format
        std_al = torch.std(attribution_maps, dim=[1, 2, 3], keepdim=True) + epsilon

        # Replace NaN values with a default value (e.g., 0) in attribution_maps, mean_al, std_al, and prior_maps
        attribution_maps = torch.where(torch.isnan(attribution_maps), torch.zeros_like(attribution_maps), attribution_maps)
        mean_al = torch.where(torch.isnan(mean_al), torch.zeros_like(mean_al), mean_al)
        std_al = torch.where(torch.isnan(std_al), torch.zeros_like(std_al), std_al)
        prior_maps = torch.where(torch.isnan(prior_maps), torch.zeros_like(prior_maps), prior_maps)

        # Calculate the PAL loss
        # Ensure that the broadcasting in the subtraction and division is correct
        pal_loss = -torch.sum((attribution_maps - mean_al) / std_al * prior_maps, dim=[1, 2, 3])

        # Return the mean loss over the batch
        return torch.mean(pal_loss)
