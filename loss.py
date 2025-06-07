import torch
import torch.nn as nn

class LossBalancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3)) 

    def forward(self, loss_radar, loss_bed, loss_uncertainty):
        loss = (
            torch.exp(-self.log_vars[0]) * loss_radar + self.log_vars[0] +
            torch.exp(-self.log_vars[1]) * loss_bed + self.log_vars[1] +
            torch.exp(-self.log_vars[2]) * loss_uncertainty + self.log_vars[2]
        )
        return loss

def bayesian_uncertainty_loss(model, input_data, edge_index, target, radar_mask, radar_confidence, num_samples, loss_balancer):
    """
    Hybrid loss: supervised radar MSE + fallback BedMachine MSE + uncertainty regularization.
    """
    model.train()

    preds = []
    for _ in range(num_samples):
        pred = model(input_data, edge_index).squeeze()
        preds.append(pred)

    all_preds = torch.stack(preds)  # (samples, N)
    mean_pred = torch.mean(all_preds, dim=0)
    epistemic_uncertainty = torch.var(all_preds, dim=0)

    radar_mask = radar_mask.flatten()
    radar_confidence = radar_confidence.flatten()
    target = target.flatten()

    radar_indices = radar_mask
    non_radar_indices = ~radar_mask

    loss_radar = (
        torch.mean(((mean_pred[radar_indices] - target[radar_indices]) ** 2) * radar_confidence[radar_indices])
        if radar_indices.sum() > 0
        else torch.tensor(0.0, device=mean_pred.device)
    )

    loss_bedmachine = (
        torch.mean((mean_pred[non_radar_indices] - target[non_radar_indices]) ** 2)
        if non_radar_indices.sum() > 0
        else torch.tensor(0.0, device=mean_pred.device)
    )

    loss_uncertainty = torch.mean(epistemic_uncertainty)

    total_loss = loss_balancer(loss_radar, loss_bedmachine, loss_uncertainty)
    return total_loss