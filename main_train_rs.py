from torch.utils.data import DataLoader
import h5py
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from gradient_covariates import gradient_covariates
from trend_surface import trend_surface
from compute_radar_confidence import compute_radar_confidence
from grid_to_graph import grid_to_graph
from BedTopoDataset import BedTopoDataset
from model import BedTopoGCN
from loss import bayesian_uncertainty_loss
from loss import LossBalancer


import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    variables = ['surf_SMB', 'surf_dhdt', 'surf_elv', 'surf_vx', 'surf_vy', 'surf_x', 'surf_y']
    data_dict = {}
    with h5py.File('./data/hackathon.h5', 'r') as f:
        for var in variables:
            data_dict[var] = np.flipud(np.transpose(f[var][0:600, 600:1200])).copy()   

    input_variables = ['surf_SMB', 'surf_dhdt', 'surf_elv', 'surf_vx', 'surf_vy']
    inputs = np.stack([data_dict[var] for var in input_variables], axis=0)

    gradient_features = gradient_covariates(inputs)
    inputs = np.vstack([inputs, gradient_features])

    trend_features = []
    for i in range(inputs.shape[0]):
        trend = trend_surface(inputs[i])
        trend_features.append(trend)

    trend_features = np.vstack(trend_features)
    inputs = np.vstack([inputs, trend_features])

    with h5py.File('./dataset/bed_BedMachine.h5', 'r') as file:
        bedmachine_data = np.flipud(np.transpose(file['bed_BedMachine'][0:600, 600:1200])).copy() 
    target_bed = bedmachine_data

    mean_target = np.mean(target_bed)
    std_target = np.std(target_bed)
    target_bed = (target_bed - mean_target) / std_target

    surf_x_min = np.min(np.abs(data_dict['surf_x']))
    surf_y_min = np.min(np.abs(data_dict['surf_y']))
    radar_mask = np.zeros(target_bed.shape, dtype=bool)

    full_data_df = pd.read_csv('./dataset/data_full.csv')
    for _, row in full_data_df.iterrows():
        x_idx = int(600 - np.round((np.abs(row['surf_x']) - surf_x_min) / 150) - 1)
        y_idx = int(np.round((np.abs(row['surf_y']) - surf_y_min) / 150))
        if 0 <= x_idx < 600 and 0 <= y_idx < 600:
            radar_mask[x_idx, y_idx] = True

    radar_mask_tensor = torch.tensor(radar_mask, dtype=torch.bool)
    radar_confidence = compute_radar_confidence(radar_mask)
    radar_confidence_tensor = torch.tensor(radar_confidence.flatten(), dtype=torch.float32).to(device)

    
    height, width = inputs[0].shape  
    graph_inputs = inputs.reshape(inputs.shape[0], -1).T  
    edge_index = grid_to_graph(height, width)  
    graph_target = target_bed.flatten()

    
    data = Data(x=torch.tensor(graph_inputs, dtype=torch.float32), 
            edge_index=edge_index, 
            y=torch.tensor(graph_target, dtype=torch.float32))

    
    model = BedTopoGCN(in_channels=data.x.shape[1], hidden_channels=128, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = BedTopoDataset(inputs, target_bed, radar_mask_tensor, patch_size=16, stride=8)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    dataset = BedTopoDataset(inputs, target_bed, radar_mask_tensor, patch_size=16, stride=8)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    best_loss = float('inf')
    patience = 5000
    counter = 0
    num_epochs = 20000
    num_samples = 10
    loss_balancer = LossBalancer().to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        # for input_patch, target_patch, radar_mask_patch, _ in train_loader: ### for horizontal slicing
        for input_patch, target_patch, radar_mask_patch in train_loader:
            input_patch = input_patch.to(device)
            target_patch = target_patch.flatten(start_dim=1).to(device)
            
            graph_inputs = input_patch.permute(0, 2, 3, 1).reshape(-1, input_patch.shape[1])
        
            patch_size = input_patch.shape[2]
            edge_index = grid_to_graph(patch_size, patch_size).to(device)

            optimizer.zero_grad()

            # Compute radar confidence for the patch
            patch_radar_confidence = radar_confidence_tensor[:graph_inputs.size(0)]

            loss = bayesian_uncertainty_loss(
                model, graph_inputs, edge_index, target_patch.flatten(), 
                radar_mask_patch.to(device), patch_radar_confidence, 
                num_samples=num_samples, loss_balancer=loss_balancer
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()  # Disable dropout for standard validation predictions
        val_loss = 0.0
        with torch.no_grad():
            # for input_patch, target_patch, radar_mask_patch, _ in val_loader: ### for horizontal slicing
            for input_patch, target_patch, radar_mask_patch in train_loader:
                input_patch = input_patch.to(device)
                target_patch = target_patch.flatten(start_dim=1).to(device)

                # Reshape input_patch for GAT
                graph_inputs = input_patch.permute(0, 2, 3, 1).reshape(-1, input_patch.shape[1])
                outputs = model(graph_inputs, edge_index.to(device)).squeeze()

                # Compute radar confidence for the patch
                patch_radar_confidence = radar_confidence_tensor[:graph_inputs.size(0)]

                loss = bayesian_uncertainty_loss(
                    model, graph_inputs, edge_index, target_patch.flatten(), 
                    radar_mask_patch.to(device), patch_radar_confidence, 
                    num_samples=num_samples, loss_balancer=loss_balancer
                )

            val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping and model saving
        if val_loss < best_loss:
            print(f"Validation loss improved from {best_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_loss = val_loss
            torch.save(model.state_dict(), './saved_models/best_bayesian_uncertainty_gcn_model_top_right.pth')
            counter = 0
        else:
            counter += 1
            print(f"No improvement for {counter} epochs.")
        if counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

if __name__ == "__main__":
    main()