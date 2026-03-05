import torch
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
import torchvision.transforms as transforms # Assuming this is where your transforms are
import json
# Define a default max pixel value if not passed.
# This should ideally match what your process_xray used as a multiplier.
DEFAULT_MAX_PIXEL_VALUE = 65535.0

class KneeMILDataset(Dataset):
    def __init__(self, h5_file_path, sample_group_names_list, transform=None,
                 kl_grade_mapping=None, max_pixel_value=DEFAULT_MAX_PIXEL_VALUE, check_patches=False): # Added max_pixel_value
        """
        Args:
            h5_file_path (str): Path to the HDF5 file.
            sample_group_names_list (list): List of group names (e.g., "patient_001_L") for this dataset split.
            transform (callable, optional): Optional transform to be applied on a patch tensor [0,1] range.
            kl_grade_mapping (dict, optional): Mapping for KL grades.
            max_pixel_value (float, optional): The maximum possible pixel value in the raw patch data
                                               (e.g., 65535.0 if process_xray scaled to that).
                                               Used for scaling patches to [0,1].
        """
        self.h5_file_path = h5_file_path
        self.sample_group_names = sample_group_names_list
        self.transform = transform
        self.kl_grade_mapping = kl_grade_mapping
        self.max_pixel_value = float(max_pixel_value) # Ensure it's a float for division

        if check_patches:
            print("[Dataset] Checking patch integrity...")
            with h5py.File(self.h5_file_path, 'r') as hf:
                for g in sample_group_names_list:
                    try:
                        if g not in hf:
                            print(f"  - DROP {g}: missing group in HDF5")
                            continue

                        patches = hf[g]['patches']

                        # ---- Check 1: empty ----
                        if patches.shape[0] == 0:
                            print(f"  - DROP {g}: empty patch bag")
                            continue

                        # ---- Check 2: patch shape consistency ----
                        shapes = []
                        for i in range(min(patches.shape[0], 5)):  # sample 5 patches
                            shapes.append(hf[g]['patches'][i].shape)

                        if len(set(shapes)) != 1:
                            print(f"  - DROP {g}: inconsistent patch shapes: {set(shapes)}")
                            continue

                        # ---- Check 3: aux feature existence ----
                        if 'aux_feature' not in hf[g]:
                            print(f"  - DROP {g}: no aux_feature field")
                            continue

                        # If passed all checks → keep
                        self.sample_group_names.append(g)

                    except Exception as e:
                        print(f"  - DROP {g}: exception during checking → {e}")

            print(f"[Dataset] Valid: {len(self.sample_group_names)} / {len(sample_group_names_list)} cases")

        else:
            # If checking disabled, just accept everything
            self.sample_group_names = sample_group_names_list
        
        with open("clean_samples.json", "w") as f:
            json.dump(self.sample_group_names, f)
    def __getitem__(self, idx):
        group_name = self.sample_group_names[idx]

        with h5py.File(self.h5_file_path, 'r') as hf:
            # Patches are loaded as np.float32, shape (N_patches, H, W, C), range [0, max_pixel_value]
            patches_data = hf[group_name]['patches'][:]
            kl_grade_val = hf[group_name]['kl_grade'][0]
            aux_feature = hf[group_name]['aux_feature'][:] # Convert to numpy array

            if self.kl_grade_mapping:
                kl_grade_val = self.kl_grade_mapping.get(kl_grade_val, kl_grade_val)

        processed_patches_list = [] # Use a different name to avoid confusion with 'patches_data'

        if patches_data.shape[0] == 0:
            print(f"Warning: Empty patches for {group_name} in __getitem__. Returning dummy data.")
            # Create a dummy tensor that matches expected output shape after transform
            # If transform is None, it's harder to know the exact final shape.
            # For now, let's assume C=1, H=16, W=16.
            # This dummy patch will also be [0,1] scaled.
            dummy_patch_tensor = torch.zeros((1, 16, 16), dtype=torch.float32)
            if self.transform:
                # The transform now expects a [0,1] tensor
                dummy_patch_tensor = self.transform(dummy_patch_tensor)
            # Return as a list of one dummy patch for collate_fn compatibility
            return [dummy_patch_tensor], torch.tensor(kl_grade_val, dtype=torch.long)

        for i in range(patches_data.shape[0]):
            patch_np = patches_data[i]  # Shape (H, W, C), np.float32, range [0, max_pixel_value]

            # 1. Scale to [0, 1.0]
            patch_np_scaled = patch_np / self.max_pixel_value
            # Ensure clipping if any minor floating point issues exceed 1.0, though unlikely with direct division
            patch_np_scaled = np.clip(patch_np_scaled, 0.0, 1.0)


            # 2. Transpose to (C, H, W) for PyTorch
            patch_np_transposed = np.transpose(patch_np_scaled, (2, 0, 1))

            # 3. Convert to PyTorch float tensor
            patch_tensor = torch.from_numpy(patch_np_transposed).float() # Already float, but good practice

            # 4. Apply transformations (which now expect a [0,1] float tensor)
            if self.transform:
                patch_tensor = self.transform(patch_tensor)

            processed_patches_list.append(patch_tensor)

        return processed_patches_list, torch.tensor(kl_grade_val, dtype=torch.long), group_name, torch.tensor(aux_feature, dtype=torch.float)

    def __len__(self):
        return len(self.sample_group_names)



# def mil_collate_fn(batch):
#     """
#     Collate function for MIL where each sample has a variable number of patches.
#     Args:
#         batch: A list of tuples, where each tuple is (list_of_patch_tensors, label_tensor)
#     Returns:
#         A tuple (list_of_patch_bags, labels_batch_tensor)
#         - list_of_patch_bags: A list where each element is a tensor of patches for a sample (N_i, C, H, W)
#         - labels_batch_tensor: A tensor of labels (batch_size)
#     """
#     patch_bags = []
#     labels = []

#     for item_patches, item_label in batch:
#         if not item_patches: # Handle case where a sample might have 0 patches after all
#             print("Warning: mil_collate_fn encountered an empty patch list for a sample.")
#             # Option 1: Skip this sample (might cause issues with batch size)
#             # continue
#             # Option 2: Create a dummy patch bag (e.g., one zero patch)
#             # This depends on how the model handles it. For now, let's try to ensure non-empty.
#             # This should be rare if Dataset filtering is good.
#             # If we must proceed, a dummy tensor might be needed:
#             # dummy_bag = torch.zeros((1, 1, TARGET_PATCH_SIZE[0], TARGET_PATCH_SIZE[1])) # N, C, H, W
#             # patch_bags.append(dummy_bag)
#             # labels.append(item_label)
#             continue # Simplest for now: skip problematic samples if they slip through

#         # Stack patches for the current sample into a single tensor (N_i, C, H, W)
#         # item_patches is already a list of (C,H,W) tensors
#         if isinstance(item_patches, list) and len(item_patches) > 0:
#              current_bag = torch.stack(item_patches, dim=0)
#              patch_bags.append(current_bag)
#              labels.append(item_label)
#         elif torch.is_tensor(item_patches) and item_patches.ndim == 4: # Already a stacked bag
#              patch_bags.append(item_patches)
#              labels.append(item_label)


#     if not labels: # If all samples in batch were skipped
#         return None, None # Or handle appropriately

#     labels_batch_tensor = torch.stack(labels, dim=0)
#     return patch_bags, labels_batch_tensor # patch_bags is a list of tensors


# def mil_collate_fn(batch):
#     patch_bags = []
#     labels = []
#     ids = []

#     for item in batch:
#         if len(item) == 3:
#             item_patches, item_label, item_id = item
#         else:
#             raise ValueError("Each sample in batch must be a (patches, label, id) tuple.")

#         if not item_patches:
#             print(f"Warning: Empty patch list for ID {item_id}")
#             continue

#         if isinstance(item_patches, list) and len(item_patches) > 0:
#             current_bag = torch.stack(item_patches, dim=0)
#         elif torch.is_tensor(item_patches) and item_patches.ndim == 4:
#             current_bag = item_patches
#         else:
#             continue

#         patch_bags.append(current_bag)
#         labels.append(item_label)
#         ids.append(item_id)

#     if not labels:
#         return None, None, None

#     labels_batch_tensor = torch.stack(labels, dim=0)
#     return patch_bags, labels_batch_tensor, ids

# Add feature
def mil_collate_fn(batch):
    patch_bags = []
    labels = []
    ids = []
    features = []

    for item in batch:
        if len(item) == 4:
            item_patches, item_label, item_id, item_feature = item
        else:
            raise ValueError("Each sample in batch must be a (patches, label, id, feature) tuple.")

        if not item_patches:
            print(f"Warning: Empty patch list for ID {item_id}")
            continue

        # Handle patch bag format
        if isinstance(item_patches, list) and len(item_patches) > 0:
            current_bag = torch.stack(item_patches, dim=0)
        elif torch.is_tensor(item_patches) and item_patches.ndim == 4:
            current_bag = item_patches
        else:
            continue

        patch_bags.append(current_bag)
        labels.append(item_label)
        ids.append(item_id)
        features.append(item_feature)

    if not labels:
        return None, None, None, None

    labels_batch_tensor = torch.stack(labels, dim=0)
    features_batch_tensor = torch.stack(features, dim=0)

    return patch_bags, labels_batch_tensor, ids, features_batch_tensor
