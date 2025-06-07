from torch.utils.data import Dataset
import torch


class BedTopoDataset(Dataset):
    def __init__(self, inputs, target, radar_mask, patch_size=16, stride=8, slice_axis='horizontal', num_slices = 30):  
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.radar_mask = radar_mask
        self.patch_size = patch_size
        self.stride = stride
        self.slice_axis = slice_axis
        self.num_slices = num_slices
        self.height, self.width = self.target.shape
        self.slice_size = self.height // self.num_slices if self.slice_axis == 'horizontal' else self.width // self.num_slices
        self.patches = self._generate_patches()

    def _get_slice_index(self, row, col):
        if self.slice_axis == 'horizontal':
            return row // self.slice_size
        else:
            return col // self.slice_size

    def _generate_patches(self):
        patches = []
        for i in range(0, self.height - self.patch_size + 1, self.stride):
            for j in range(0, self.width - self.patch_size + 1, self.stride):
                slice_idx = self._get_slice_index(i, j)
                patches.append((i, j, slice_idx))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        row, col, slice_idx = self.patches[idx]
        input_patch = self.inputs[:, row:row+self.patch_size, col:col+self.patch_size]
        target_patch = self.target[row:row+self.patch_size, col:col+self.patch_size]
        radar_mask_patch = self.radar_mask[row:row+self.patch_size, col:col+self.patch_size]
        return input_patch, target_patch, radar_mask_patch, slice_idx