import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models # For using pre-trained backbones
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import copy

# normalize the attention map and 
# change multiplying from patches into feature extractor output
# add the raw feature and enhanced feature together for aggregator
import matplotlib.pyplot as plt
range1 = np.arange(9, 27)  # Indices 9 to 26
range2 = np.arange(44, 67) # Indices 44 to 66
PATCH_POINT_INDICES = np.concatenate([range1, range2])

def plot_patches_grid_with_heatmaps(
    IMG_SAVE_PATH,
    patches_list,
    heatmaps_list,
    patch_indices_map,
    num_cols=8,
    figure_title="",
    patch_cmap='gray',
    heatmap_cmap=plt.get_cmap('Reds'),
    base_heatmap_alpha=0.3,
    attention_scores_norm=None,
    title_fontsize=8,
    interpolation_method='nearest',
    img_type="patch_dot_attentionmap_stacked"
):
    import math
    import os
    import matplotlib.pyplot as plt

    if len(patches_list) == 0:
        print("No patches to display.")
        return

    n_patches = len(patches_list)
    if n_patches != len(heatmaps_list) or n_patches != len(patch_indices_map):
        print("Error: Mismatch in lengths of patches, heatmaps, or patch_indices_map.")
        return

    n_rows = math.ceil(n_patches / num_cols)

    # 如果是 tensor → 轉成 numpy 並計算全域 min/max
    if hasattr(patches_list, "min"):  # torch tensor
        global_min = patches_list.min().item()
        global_max = patches_list.max().item()
    else:  # numpy array
        global_min = patches_list.min()
        global_max = patches_list.max()

    patches_list = patches_list - global_min  # shift so min = 0

    # 建立子圖
    fig, axes = plt.subplots(n_rows, num_cols, figsize=(num_cols * 2, n_rows * 2.2))
    axes = axes.flatten() if n_rows * num_cols > 1 else [axes]

    img_handle = None  # 用於 colorbar

    for idx in range(n_patches):
        ax = axes[idx]
        patch = patches_list[idx]
        heatmap = heatmaps_list[idx]

        # 如果是 tensor → numpy
        if hasattr(patch, "cpu"):
            patch_np = patch.cpu().numpy().squeeze()
        else:
            patch_np = patch.squeeze()

        img_handle = ax.imshow(
            patch_np, cmap=patch_cmap,
            interpolation=interpolation_method,
            vmin=patches_list.min(), vmax=patches_list.max()
        )

        current_alpha = base_heatmap_alpha
        if attention_scores_norm is not None and idx < len(attention_scores_norm):
            current_alpha *= attention_scores_norm[idx]
            current_alpha = max(0.0, min(1.0, current_alpha))

        # 疊加 heatmap（如需要）
        # ax.imshow(heatmap, cmap=heatmap_cmap, alpha=current_alpha, interpolation=interpolation_method)

        ax.set_title(f"Point {patch_indices_map[idx]}", fontsize=title_fontsize)
        ax.axis('off')

    for idx in range(n_patches, n_rows * num_cols):
        axes[idx].axis('off')

    # 在右側加 colorbar，不擋圖
    if img_handle is not None:
        fig.subplots_adjust(right=0.88)  # 預留右側空間
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(img_handle, cax=cbar_ax)
        cbar.set_label('Pixel Intensity', fontsize=title_fontsize)

    if figure_title:
        fig.suptitle(figure_title, fontsize=title_fontsize + 4, y=0.99)

    plt.savefig(os.path.join(IMG_SAVE_PATH, f"heatmap_{img_type}.png"), format='png')
    plt.close(fig)

def plot_patches_grid_for_attscore(
    IMG_SAVE_PATH,
    patches_list,
    heatmaps_list,
    patch_indices_map,
    num_cols=8,
    figure_title="",
    patch_cmap='gray',
    heatmap_cmap=plt.get_cmap('Reds'),
    base_heatmap_alpha=0.3,
    attention_scores_norm=None, # Optional: for alpha modulation
    title_fontsize=8,
    interpolation_method='nearest',
    img_type="patch_dot_attentionmap_stacked" # Added to differentiate image types
):
    
    print("plot fig")
    """
    Displays a grid of image patches with overlaid heatmaps.

    Args:
        patches_list (list/np.array): List of image patches (each a NumPy array).
        heatmaps_list (list/np.array): List of heatmaps corresponding to patches.
        patch_indices_map (list/np.array): List of original indices/labels for titles.
        num_cols (int): Number of columns in the grid.
        figure_title (str): Overall title for the figure.
        patch_cmap (str): Colormap for the base patches.
        heatmap_cmap (str): Colormap for the heatmaps.
        base_heatmap_alpha (float): Base alpha transparency for heatmaps.
        attention_scores_norm (list/np.array, optional): Normalized attention scores.
                                                        If provided, heatmap_alpha = base_heatmap_alpha * score.
        title_fontsize (int): Font size for individual patch titles.
        interpolation_method (str): Interpolation method for imshow.
    """
    import math
    import os

    if len(patches_list) == 0:
        print("No patches to display.")
        return

    n_patches = len(patches_list)
    if n_patches != len(heatmaps_list) or n_patches != len(patch_indices_map):
        print("Error: Mismatch in lengths of patches, heatmaps, or patch_indices_map.")
        return

    n_rows = math.ceil(n_patches / num_cols)

    fig, axes = plt.subplots(n_rows, num_cols, figsize=(num_cols * 2, n_rows * 2.2)) # Slightly taller for titles
    
    # Flatten axes array for easier indexing, handling single row/col cases
    if n_rows * num_cols > 1:
        axes = axes.flatten()
    elif n_rows * num_cols == 1: # Single subplot
        axes = [axes] 
    else: # No subplots if n_patches is 0 (already handled)
        return
    
    global_min = patches_list.min()
    global_max = patches_list.max()
    range_eps = global_max - global_min + 1e-8
    # Before: patch_np = patch.squeeze()
    # patch_np = (patches_list.squeeze() - global_min) / range_eps


    for idx in range(n_patches):
        ax = axes[idx]
        patch = patches_list[idx]
        heatmap = heatmaps_list[idx]

        # Prepare patch (squeeze to 2D if needed)
        # patch_np = patch.squeeze()

        # 全域 min/max normalize
        patch_np = (patch.squeeze() - global_min) / range_eps
        # Plot base patch
        ax.imshow(patch_np, cmap=patch_cmap, interpolation=interpolation_method)

        # Determine heatmap alpha
        current_alpha = base_heatmap_alpha
        if attention_scores_norm is not None and idx < len(attention_scores_norm):
            current_alpha *= attention_scores_norm[idx]
            current_alpha = np.clip(current_alpha, 0.0, 1.0) # Ensure alpha is valid

        # Overlay heatmap
        # ax.imshow(heatmap, cmap=heatmap_cmap, alpha=current_alpha, interpolation=interpolation_method)

        # Set title and turn off axis
        ax.set_title(f"Point {patch_indices_map[idx]}", fontsize=title_fontsize)
        ax.axis('off')

    # Hide any unused subplots
    for idx in range(n_patches, n_rows * num_cols):
        axes[idx].axis('off')

    if figure_title:
        fig.suptitle(figure_title, fontsize=title_fontsize + 4, y=0.99) # Adjust y to prevent overlap

    plt.tight_layout(rect=[0, 0, 1, 0.95 if figure_title else 0.98]) # Adjust rect for suptitle
    # plt.show()
    # plt.savefig(os.path.join(IMG_SAVE_PATH, f"heatmap_{img_type}.eps"), format='eps')
    plt.savefig(os.path.join(IMG_SAVE_PATH, f"heatmap_{img_type}.png"), format='png')



class PatchFeatureExtractor(nn.Module):
    def __init__(self, output_embedding_dim=128, num_channels=32, dropout_rate=0.4, adaptive_pool_output_size=(2, 2)):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels), # ADDED
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels), # ADDED
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels), # ADDED
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16 -> 8x8
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels*2), # ADDED
            nn.ReLU(),
            nn.Conv2d(num_channels*2, num_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels*2), # ADDED
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x8 -> 4x4
        )
        self.conv_block3 = nn.Sequential( # ADDED
            nn.Conv2d(num_channels*2, num_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 4x4 -> 2x2
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        fc_input_size = (num_channels * 4) * 2 * 2 # Recalculate based on 2x2 output
        self.fc = nn.Linear(fc_input_size, output_embedding_dim)

    def forward(self, x_patch): # x_patch is (C, H, W)
        x = self.conv_block1(x_patch)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        embedding = self.fc(x)
        embedding = self.dropout(embedding)
        return embedding


class MILAggregator(nn.Module):
    def __init__(self, input_embedding_dim, aggregation_type='attention', dropout_rate=0.4):
        super().__init__()
        self.input_embedding_dim = input_embedding_dim
        self.aggregation_type = aggregation_type

        if self.aggregation_type == 'attention':
            # Attention mechanism (a simple version)
            self.attention_V = nn.Linear(input_embedding_dim, 128)
            self.attention_U = nn.Linear(input_embedding_dim, 128)
            self.attention_w = nn.Linear(128, 1) # Outputs one score per patch embedding
        elif self.aggregation_type not in ['mean', 'max']:
            raise ValueError("Unsupported aggregation type")

        self.dropout = nn.Dropout(p=dropout_rate) # ADDED

    def forward(self, patch_embeddings_bag): # patch_embeddings_bag is (N_patches, embedding_dim)
        if patch_embeddings_bag.nelement() == 0: # Handle empty bag
            # Return zeros or some other placeholder prediction
            # This implies the bag itself might be problematic
            print("Warning: MILAggregator received an empty patch_embeddings_bag.")
            return torch.zeros((1, self.num_classes), device=patch_embeddings_bag.device)


        if self.aggregation_type == 'mean':
            aggregated_features = torch.mean(patch_embeddings_bag, dim=0) # (embedding_dim)
        elif self.aggregation_type == 'max':
            aggregated_features, _ = torch.max(patch_embeddings_bag, dim=0) # (embedding_dim)
        elif self.aggregation_type == 'attention':
            # A = H (N_patches, embedding_dim)
            A_V = torch.tanh(self.attention_V(patch_embeddings_bag)) # (N_patches, 128)
            A_U = torch.sigmoid(self.attention_U(patch_embeddings_bag)) # (N_patches, 128) # Gating
            att_scores_unnorm = self.attention_w(A_V * A_U) # (N_patches, 1) element-wise product
            att_scores_unnorm = self.dropout(att_scores_unnorm) # Option 2: Dropout scores before softmax
            att_scores = F.softmax(att_scores_unnorm, dim=0) # (N_patches, 1)

            # Weighted sum of patch embeddings
            aggregated_features = torch.sum(att_scores * patch_embeddings_bag, dim=0) # (embedding_dim)
        else: # Should not happen due to init check
            raise ValueError("Unsupported aggregation type")

        # Add batch dimension if it was squeezed by mean/max/sum over dim=0
        if aggregated_features.ndim == 1:
            aggregated_features = aggregated_features.unsqueeze(0) # (1, embedding_dim)

        aggregated_features = self.dropout(aggregated_features) # APPLY DROPOUT
        return att_scores, aggregated_features # ADDED



# Use the code from https://github.com/imedslab/KneeOARSIGrading
class GlobalWeightedAveragePooling(nn.Module):
    """
    "Global Weighted Average Pooling Bridges Pixel-level Localization and Image-level Classiﬁcation".

    Class-agnostic version.

    """

    def __init__(self, n_feats, use_hidden=False):
        super().__init__()
        if use_hidden:
            self.conv = nn.Sequential(
                nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(n_feats),
                nn.ReLU(True),
                nn.Dropout2d(0.5),
                nn.Conv2d(n_feats, 1, kernel_size=1, bias=True)
            )
        else:
            self.conv = nn.Conv2d(n_feats, 1, kernel_size=1, bias=True)

    def fscore(self, x: torch.Tensor):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3])
        return x

# Use the code from https://github.com/imedslab/KneeOARSIGrading
class ClassificationHead(nn.Module):
    def __init__(self, n_features, n_cls, use_bnorm=True, drop=0.5, use_gwap=False,
                 use_gwap_hidden=False, no_pool=False):
        super(ClassificationHead, self).__init__()

        clf_layers = []
        if use_bnorm:
            clf_layers.append(nn.BatchNorm1d(n_features))

        if drop > 0:
            clf_layers.append(nn.Dropout(drop))

        clf_layers.append(nn.Linear(n_features, n_cls))

        self.classifier = nn.Sequential(*clf_layers)
        self.no_pool = no_pool

        if use_gwap:
            self.gwap = GlobalWeightedAveragePooling(n_features, use_hidden=use_gwap_hidden)

    def forward(self, o):
        if not self.no_pool:
            if not hasattr(self, 'gwap'):
                avgp = F.adaptive_avg_pool2d(o, 1).view(o.size(0), -1)
            else:
                avgp = self.gwap(o).view(o.size(0), -1)
        else:
            avgp = o
        clf_result = self.classifier(avgp)
        return clf_result

# Use the code from https://github.com/imedslab/KneeOARSIGrading
from typing import Tuple
class MultiTaskHead(nn.Module):
    def __init__(self, n_feats, n_tasks, n_cls: int or Tuple[int], clf_bnorm, dropout, use_gwap=False,
                 use_gwap_hidden=False, no_pool=False):

        super(MultiTaskHead, self).__init__()

        if isinstance(n_cls, int):
            n_cls = (n_cls, )

        if isinstance(n_tasks, int):
            n_tasks = (n_tasks,)

        assert len(n_cls) == len(n_tasks)

        self.n_tasks = n_tasks
        self.n_cls = n_cls

        for task_type_idx, (n_tasks, task_n_cls) in enumerate(zip(self.n_tasks, self.n_cls)):
            for task_idx in range(n_tasks):
                self.__dict__['_modules'][f'head_{task_type_idx+task_idx}'] = nn.Linear(n_feats, task_n_cls)    

    def forward(self, features):
        res = []
        for j in range(sum(self.n_tasks)):
            res.append(self.__dict__['_modules'][f'head_{j}'](features))
        return res
    
class OrdinalHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes - 1)  # K-1 個logit
        )
        
    def forward(self, x):
        logits = self.fc(x)          # (B, K-1)
        prob = torch.sigmoid(logits) # 轉成 >=k 的機率
        return logits, prob
    
class Ordinal_MultiTaskHead(nn.Module):
    def __init__(self, n_feats, n_tasks, n_cls: int or Tuple[int], clf_bnorm, dropout, use_gwap=False,
                 use_gwap_hidden=False, no_pool=False):

        super(Ordinal_MultiTaskHead, self).__init__()

        if isinstance(n_cls, int):
            n_cls = (n_cls, )

        if isinstance(n_tasks, int):
            n_tasks = (n_tasks,)

        assert len(n_cls) == len(n_tasks)

        self.n_tasks = n_tasks
        self.n_cls = n_cls

        for task_type_idx, (n_tasks, task_n_cls) in enumerate(zip(self.n_tasks, self.n_cls)):
            for task_idx in range(n_tasks):
                self.__dict__['_modules'][f'head_{task_type_idx+task_idx}'] = OrdinalHead(n_feats, task_n_cls)    

    def forward(self, features):
        res = []
        for j in range(sum(self.n_tasks)):
            out, prob = self.__dict__['_modules'][f'head_{j}'](features)
            # print(out.shape, prob.shape)
            res.append(out)
        return res
    
class GlobalWeightedAveragePooling(nn.Module):
    """
    Learnable Global Weighted Average Pooling
    input: [B, C, H, W]
    output: [B, C]
    """
    def __init__(self, in_channels):
        super(GlobalWeightedAveragePooling, self).__init__()
        self.attention = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()
        weight = self.attention(x)  
        weight = F.softmax(weight.view(B, -1), dim=-1).view(B, 1, H, W)  # normalize
        out = (x * weight).sum(dim=[2, 3])  # [B, C]
        return out


class GlobalWeightedMaxPooling(nn.Module):
    """
    Learnable Global Weighted Max Pooling
    input: [B, C, H, W]
    output: [B, C]
    """
    def __init__(self, in_channels):
        super(GlobalWeightedMaxPooling, self).__init__()
        self.attention = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()
        weight = self.attention(x)
        weight = F.softmax(weight.view(B, -1), dim=-1).view(B, 1, H, W)
        out = (x * weight).amax(dim=[2, 3])  # [B, C]
        return out

class GlobalWeightedAveragePooling1D(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x):  # x: [N, D]
        weights = torch.softmax(self.attention(x), dim=0)  # [N, 1]
        out = torch.sum(x * weights, dim=0, keepdim=True)  # [1, D]
        return out

class GlobalWeightedMaxPooling1D(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x):  # x: [N, D]
        weights = torch.softmax(self.attention(x), dim=0)  # [N, 1]
        weighted_x = x * weights  # [N, D]
        out, _ = torch.max(weighted_x, dim=0, keepdim=True)  # [1, D]
        return out

class CompleteMILModel(nn.Module):
    def __init__(self, feature_extractor_out_dim, num_classes, aggregation_type='attention'):
        super().__init__()
        self.patch_feature_extractor = PatchFeatureExtractor(output_embedding_dim=feature_extractor_out_dim)
        self.aggregator = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        aggregation_type=aggregation_type)
        self.classifier = nn.Linear(feature_extractor_out_dim, num_classes) 

    def forward(self, list_of_patch_bags): # list_of_patch_bags: list of tensors, each (N_i, C, H, W)
        batch_logits = []
        batch_att_scores = []
        aggregated_features_all = []
        patch_embeddings_stacked_all = []

        for patch_bag_tensor in list_of_patch_bags: # Iterate through samples in the batch
            # patch_bag_tensor is (N_i, C, H, W) for the i-th sample in batch
            if patch_bag_tensor.nelement() == 0 or patch_bag_tensor.shape[0] == 0:
                print("Warning: CompleteMILModel encountered an empty patch_bag_tensor for a sample.")
                # Create dummy logits for this problematic sample
                # This ensures the batch processing continues, but this sample won't contribute meaningfully
                dummy_logit = torch.zeros((1, self.aggregator.num_classes), device=next(self.parameters()).device)
                batch_logits.append(dummy_logit)
                continue

            # Process each patch in the bag
            # Need to pass patches one by one or as a batch to feature extractor
            # If feature_extractor takes (B, C, H, W), then patch_bag_tensor is already that.
            patch_embeddings_stacked = self.patch_feature_extractor(patch_bag_tensor) # (N_i, embedding_dim)
            patch_embeddings_stacked_all.append(patch_embeddings_stacked.cpu()) # Store for later use

            # Aggregate and classify
            att_scores, aggregated_features = self.aggregator(patch_embeddings_stacked) # (1, num_classes)
            sample_logits = self.classifier(aggregated_features)
            batch_logits.append(sample_logits)
            batch_att_scores.append(att_scores)
            aggregated_features_all.append(aggregated_features.cpu())
            

        if not batch_logits: # If entire batch was problematic
            print("Warning: Entire batch resulted in no logits.")
            # This is a more severe issue, might need to return None or handle upstream
            # For now, return dummy based on expected batch size (though hard to know here)
            return torch.empty(0, self.aggregator.num_classes, device=next(self.parameters()).device)


        final_batch_logits = torch.cat(batch_logits, dim=0) # (batch_size, num_classes)
        final_batch_att_scores = torch.cat(batch_att_scores, dim=0) # (batch_size, num_patches)
        final_aggregated_features = torch.cat(aggregated_features_all, dim=0) # (batch_size, embedding_dim)
        patch_embeddings_stacked_all = torch.stack(patch_embeddings_stacked_all, dim=0) # (batch_size, N_i, embedding_dim)

        return final_batch_logits, final_batch_att_scores, patch_embeddings_stacked_all, final_aggregated_features 

class CompleteMILModel_MultiTask_SharedHead(nn.Module):
    def __init__(self, feature_extractor_out_dim, oai_task_num_classes, aggregation_type='attention'):
        super().__init__()
        self.oai_task_num_classes = oai_task_num_classes
        self.patch_feature_extractor = PatchFeatureExtractor(output_embedding_dim=feature_extractor_out_dim)
        self.aggregator = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        aggregation_type=aggregation_type)

        self.shared_head = nn.Linear(feature_extractor_out_dim, sum([num_cls-1 for num_cls in oai_task_num_classes.values()]))
        # self.oai_heads = nn.ModuleDict()
        # if oai_task_num_classes is not None:
        #     for task_name, num_cls in oai_task_num_classes.items():
        #         # Each ordinal head outputs K-1 logits
        #         self.oai_heads[task_name] = nn.Linear(feature_extractor_out_dim, num_cls - 1)

    def forward(self, list_of_patch_bags): # list_of_patch_bags: list of tensors, each (N_i, C, H, W)
        batch_logits = []
        batch_att_scores = []
        aggregated_features_all = []
        patch_embeddings_stacked_all = []
        oai_preds_all = {task: [] for task, _ in self.oai_task_num_classes.items()}

        for patch_bag_tensor in list_of_patch_bags: # Iterate through samples in the batch
            # patch_bag_tensor is (N_i, C, H, W) for the i-th sample in batch
            if patch_bag_tensor.nelement() == 0 or patch_bag_tensor.shape[0] == 0:
                print("Warning: CompleteMILModel encountered an empty patch_bag_tensor for a sample.")
                # Create dummy logits for this problematic sample
                # This ensures the batch processing continues, but this sample won't contribute meaningfully
                dummy_logit = torch.zeros((1, self.aggregator.num_classes), device=next(self.parameters()).device)
                batch_logits.append(dummy_logit)
                continue

            # Process each patch in the bag
            # Need to pass patches one by one or as a batch to feature extractor
            # If feature_extractor takes (B, C, H, W), then patch_bag_tensor is already that.
            patch_embeddings_stacked = self.patch_feature_extractor(patch_bag_tensor) # (N_i, embedding_dim)
            patch_embeddings_stacked_all.append(patch_embeddings_stacked.cpu()) # Store for later use

            # Aggregate and classify
            att_scores, aggregated_features = self.aggregator(patch_embeddings_stacked) # (1, num_classes)
            batch_att_scores.append(att_scores)
            aggregated_features_all.append(aggregated_features.cpu())

             # --- Auxiliary OAI predictions ---
            # for task_name, head in self.oai_heads.items():
            #     oai_pred = head(aggregated_features)  # (1, K-1)
            #     oai_preds_all[task_name].append(oai_pred)
            logits = self.shared_head(aggregated_features)  # (batch_size, total_num_logits)
            # split into tasks
            task_splits = torch.split(logits, [num_cls-1 for num_cls in self.oai_task_num_classes.values()], dim=1)
            oai_preds_all = {task: t for task, t in zip(self.oai_task_num_classes.keys(), task_splits)}
                

        final_batch_att_scores = torch.cat(batch_att_scores, dim=0) # (batch_size, num_patches)
        final_aggregated_features = torch.cat(aggregated_features_all, dim=0) # (batch_size, embedding_dim)
        patch_embeddings_stacked_all = torch.stack(patch_embeddings_stacked_all, dim=0) # (batch_size, N_i, embedding_dim)

        for task_name in oai_preds_all:
            oai_preds_all[task_name] = torch.cat(oai_preds_all[task_name], dim=0)  # (B, K-1)

        return oai_preds_all, final_batch_att_scores, patch_embeddings_stacked_all, final_aggregated_features 

class CompleteMILModel_MultiTask(nn.Module):
    def __init__(self, feature_extractor_out_dim, oai_task_num_classes, aggregation_type='attention'):
        super().__init__()
        self.oai_task_num_classes = oai_task_num_classes
        self.patch_feature_extractor = PatchFeatureExtractor(output_embedding_dim=feature_extractor_out_dim)
        self.aggregator = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        aggregation_type=aggregation_type)

        self.oai_heads = nn.ModuleDict()
        if oai_task_num_classes is not None:
            for task_name, num_cls in oai_task_num_classes.items():
                # Each ordinal head outputs K-1 logits
                self.oai_heads[task_name] = nn.Linear(feature_extractor_out_dim, num_cls)

    def forward(self, list_of_patch_bags): # list_of_patch_bags: list of tensors, each (N_i, C, H, W)
        batch_logits = []
        batch_att_scores = []
        aggregated_features_all = []
        patch_embeddings_stacked_all = []
        oai_preds_all = {task: [] for task in self.oai_heads}

        for patch_bag_tensor in list_of_patch_bags: # Iterate through samples in the batch
            # patch_bag_tensor is (N_i, C, H, W) for the i-th sample in batch
            if patch_bag_tensor.nelement() == 0 or patch_bag_tensor.shape[0] == 0:
                print("Warning: CompleteMILModel encountered an empty patch_bag_tensor for a sample.")
                # Create dummy logits for this problematic sample
                # This ensures the batch processing continues, but this sample won't contribute meaningfully
                dummy_logit = torch.zeros((1, self.aggregator.num_classes), device=next(self.parameters()).device)
                batch_logits.append(dummy_logit)
                continue

            # Process each patch in the bag
            # Need to pass patches one by one or as a batch to feature extractor
            # If feature_extractor takes (B, C, H, W), then patch_bag_tensor is already that.
            patch_embeddings_stacked = self.patch_feature_extractor(patch_bag_tensor) # (N_i, embedding_dim)
            patch_embeddings_stacked_all.append(patch_embeddings_stacked.cpu()) # Store for later use

            # Aggregate and classify
            att_scores, aggregated_features = self.aggregator(patch_embeddings_stacked) # (1, num_classes)
            batch_att_scores.append(att_scores)
            aggregated_features_all.append(aggregated_features.cpu())

             # --- Auxiliary OAI predictions ---
            for task_name, head in self.oai_heads.items():
                oai_pred = head(aggregated_features)  # (1, K-1)
                oai_preds_all[task_name].append(oai_pred)

                

        final_batch_att_scores = torch.cat(batch_att_scores, dim=0) # (batch_size, num_patches)
        final_aggregated_features = torch.cat(aggregated_features_all, dim=0) # (batch_size, embedding_dim)
        patch_embeddings_stacked_all = torch.stack(patch_embeddings_stacked_all, dim=0) # (batch_size, N_i, embedding_dim)

        for task_name in oai_preds_all:
            oai_preds_all[task_name] = torch.cat(oai_preds_all[task_name], dim=0)  # (B, K-1)

        return oai_preds_all, final_batch_att_scores, patch_embeddings_stacked_all, final_aggregated_features 


class CompleteMILModel_MultiTask_imedslab(nn.Module):
    def __init__(self, feature_extractor_out_dim, oai_task_num_classes, aggregation_type='attention'):
        super().__init__()
        self.oai_task_num_classes = oai_task_num_classes
        self.patch_feature_extractor = PatchFeatureExtractor(output_embedding_dim=feature_extractor_out_dim)
        self.aggregator = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        aggregation_type=aggregation_type)

        # self.oai_heads = nn.ModuleDict()
        self.oai_heads = MultiTaskHead(n_feats=feature_extractor_out_dim,
                                      n_tasks=(1, len(oai_task_num_classes)-1), 
                                      n_cls=(5, 4),
                                      clf_bnorm=True,
                                      dropout=0.5,
                                      use_gwap=False,
                                      use_gwap_hidden=False,
                                      no_pool=True)
        
        # if oai_task_num_classes is not None:
        #     for task_name, num_cls in oai_task_num_classes.items():
        #         # Each ordinal head outputs K-1 logits
        #         self.oai_heads[task_name] = nn.Linear(feature_extractor_out_dim, num_cls)

    def forward(self, list_of_patch_bags): # list_of_patch_bags: list of tensors, each (N_i, C, H, W)
        batch_logits = []
        batch_att_scores = []
        aggregated_features_all = []
        patch_embeddings_stacked_all = []

        oai_preds_all = {task: [] for task, _ in self.oai_task_num_classes.items()}
        all_task_names = list(self.oai_task_num_classes.keys())

        for patch_bag_tensor in list_of_patch_bags: # Iterate through samples in the batch
            # patch_bag_tensor is (N_i, C, H, W) for the i-th sample in batch
            if patch_bag_tensor.nelement() == 0 or patch_bag_tensor.shape[0] == 0:
                print("Warning: CompleteMILModel encountered an empty patch_bag_tensor for a sample.")
                # Create dummy logits for this problematic sample
                # This ensures the batch processing continues, but this sample won't contribute meaningfully
                dummy_logit = torch.zeros((1, self.aggregator.num_classes), device=next(self.parameters()).device)
                batch_logits.append(dummy_logit)
                continue

            # Process each patch in the bag
            # Need to pass patches one by one or as a batch to feature extractor
            # If feature_extractor takes (B, C, H, W), then patch_bag_tensor is already that.
            patch_embeddings_stacked = self.patch_feature_extractor(patch_bag_tensor) # (N_i, embedding_dim)
            patch_embeddings_stacked_all.append(patch_embeddings_stacked.cpu()) # Store for later use
            # print(patch_embeddings_stacked.shape)

            # Aggregate and classify
            att_scores, aggregated_features = self.aggregator(patch_embeddings_stacked) # (1, num_classes)
            batch_att_scores.append(att_scores)
            aggregated_features_all.append(aggregated_features.cpu())

             # --- Auxiliary OAI predictions ---
            oai_pred = self.oai_heads(aggregated_features)  # List of logits for each task
            for i in range(len(oai_pred)):
            #     oai_pred = head(aggregated_features)  # (1, K-1)
                # task_name = self.oai_task_num_classes.items()[i][0]
                # print(task_name)
                oai_preds_all[all_task_names[i]].append(oai_pred[i])
                

        final_batch_att_scores = torch.cat(batch_att_scores, dim=0) # (batch_size, num_patches)
        final_aggregated_features = torch.cat(aggregated_features_all, dim=0) # (batch_size, embedding_dim)
        # patch_embeddings_stacked_all = torch.stack(patch_embeddings_stacked_all, dim=0) # (batch_size, N_i, embedding_dim)

        try:
            patch_embeddings_stacked_all = torch.stack(patch_embeddings_stacked_all, dim=0)
        except RuntimeError:
            print("Skip batch due to inconsistent patch numbers")
            return None


        for task_name in oai_preds_all:
            oai_preds_all[task_name] = torch.cat(oai_preds_all[task_name], dim=0)  # (B, K-1)

        return oai_preds_all, final_batch_att_scores, patch_embeddings_stacked_all, final_aggregated_features 

class CompleteMILModel_wGP_MultiTask(nn.Module):
    def __init__(self, feature_extractor_out_dim, oai_task_num_classes, aggregation_type='attention'):
        super().__init__()
        self.oai_task_num_classes = oai_task_num_classes
        self.patch_feature_extractor = PatchFeatureExtractor(output_embedding_dim=feature_extractor_out_dim)
        self.aggregator = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        aggregation_type=aggregation_type)
        self.gwap = GlobalWeightedAveragePooling1D(in_dim=feature_extractor_out_dim)
        self.gwmp = GlobalWeightedMaxPooling1D(in_dim=feature_extractor_out_dim)


        self.oai_heads = nn.ModuleDict()
        if oai_task_num_classes is not None:
            for task_name, num_cls in oai_task_num_classes.items():
                # Each ordinal head outputs K-1 logits
                self.oai_heads[task_name] = nn.Linear(feature_extractor_out_dim, num_cls)

    def forward(self, list_of_patch_bags): # list_of_patch_bags: list of tensors, each (N_i, C, H, W)
        batch_logits = []
        batch_att_scores = []
        aggregated_features_all = []
        patch_embeddings_stacked_all = []
        oai_preds_all = {task: [] for task in self.oai_heads}

        for patch_bag_tensor in list_of_patch_bags: # Iterate through samples in the batch
            # patch_bag_tensor is (N_i, C, H, W) for the i-th sample in batch
            if patch_bag_tensor.nelement() == 0 or patch_bag_tensor.shape[0] == 0:
                print("Warning: CompleteMILModel encountered an empty patch_bag_tensor for a sample.")
                # Create dummy logits for this problematic sample
                # This ensures the batch processing continues, but this sample won't contribute meaningfully
                dummy_logit = torch.zeros((1, self.aggregator.num_classes), device=next(self.parameters()).device)
                batch_logits.append(dummy_logit)
                continue

            # Process each patch in the bag
            # Need to pass patches one by one or as a batch to feature extractor
            # If feature_extractor takes (B, C, H, W), then patch_bag_tensor is already that.
            patch_embeddings_stacked = self.patch_feature_extractor(patch_bag_tensor) # (N_i, embedding_dim)
            patch_embeddings_stacked_all.append(patch_embeddings_stacked.cpu()) # Store for later use

            # Aggregate and classify
            att_scores, aggregated_features = self.aggregator(patch_embeddings_stacked) # (1, num_classes)
            batch_att_scores.append(att_scores)
            aggregated_features_all.append(aggregated_features.cpu())

             # --- Auxiliary OAI predictions ---
            for task_name, head in self.oai_heads.items():
                if task_name == "kl":  # Example: use GWAP for KL task
                    oai_pool = self.gwap(aggregated_features)
                else:  # Use GWMP for other tasks
                    oai_pool = self.gwmp(aggregated_features)

                oai_pred = head(oai_pool)  # (1, K-1)
                oai_preds_all[task_name].append(oai_pred)


        final_batch_att_scores = torch.cat(batch_att_scores, dim=0) # (batch_size, num_patches)
        final_aggregated_features = torch.cat(aggregated_features_all, dim=0) # (batch_size, embedding_dim)
        patch_embeddings_stacked_all = torch.stack(patch_embeddings_stacked_all, dim=0) # (batch_size, N_i, embedding_dim)

        for task_name in oai_preds_all:
            oai_preds_all[task_name] = torch.cat(oai_preds_all[task_name], dim=0)  # (B, K-1)

        return oai_preds_all, final_batch_att_scores, patch_embeddings_stacked_all, final_aggregated_features 


class CompleteMILOrdinalModel(nn.Module):
    def __init__(self, feature_extractor_out_dim, num_classes, aggregation_type='attention'):
        super().__init__()
        self.patch_feature_extractor = PatchFeatureExtractor(output_embedding_dim=feature_extractor_out_dim)
        self.aggregator = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        aggregation_type=aggregation_type)
        
        self.classifier = nn.Linear(feature_extractor_out_dim, num_classes - 1) # for ordinal, output is num_classes - 1 logits

    def forward(self, list_of_patch_bags): # list_of_patch_bags: list of tensors, each (N_i, C, H, W)
        batch_logits = []
        batch_att_scores = []
        aggregated_features_all = []
        patch_embeddings_stacked_all = []

        for patch_bag_tensor in list_of_patch_bags: # Iterate through samples in the batch
            # patch_bag_tensor is (N_i, C, H, W) for the i-th sample in batch
            if patch_bag_tensor.nelement() == 0 or patch_bag_tensor.shape[0] == 0:
                print("Warning: CompleteMILModel encountered an empty patch_bag_tensor for a sample.")
                # Create dummy logits for this problematic sample
                # This ensures the batch processing continues, but this sample won't contribute meaningfully
                dummy_logit = torch.zeros((1, self.aggregator.num_classes), device=next(self.parameters()).device)
                batch_logits.append(dummy_logit)
                continue

            # Process each patch in the bag
            # Need to pass patches one by one or as a batch to feature extractor
            # If feature_extractor takes (B, C, H, W), then patch_bag_tensor is already that.
            patch_embeddings_stacked = self.patch_feature_extractor(patch_bag_tensor) # (N_i, embedding_dim)
            patch_embeddings_stacked_all.append(patch_embeddings_stacked.cpu()) # Store for later use

            # Aggregate and classify
            att_scores, aggregated_features = self.aggregator(patch_embeddings_stacked) # (1, num_classes)
            final_logits = self.classifier(aggregated_features)
            batch_logits.append(final_logits)
            batch_att_scores.append(att_scores)
            aggregated_features_all.append(aggregated_features.cpu())
            

        if not batch_logits: # If entire batch was problematic
            print("Warning: Entire batch resulted in no logits.")
            # This is a more severe issue, might need to return None or handle upstream
            # For now, return dummy based on expected batch size (though hard to know here)
            return torch.empty(0, self.aggregator.num_classes, device=next(self.parameters()).device)


        final_batch_logits = torch.cat(batch_logits, dim=0) # (batch_size, num_classes)
        final_batch_att_scores = torch.cat(batch_att_scores, dim=0) # (batch_size, num_patches)
        final_aggregated_features = torch.cat(aggregated_features_all, dim=0) # (batch_size, embedding_dim)
        patch_embeddings_stacked_all = torch.stack(patch_embeddings_stacked_all, dim=0) # (batch_size, N_i, embedding_dim)
        
        # print("num of logits:", final_batch_logits.shape)
        print("MIL Ordinal")
        
        return final_batch_logits, final_batch_att_scores, patch_embeddings_stacked_all, final_aggregated_features 
    

class CompleteMILOrdinal_MultiTask_Model(nn.Module):
    def __init__(self, feature_extractor_out_dim, oai_task_num_classes, aggregation_type='attention'):
        super().__init__()
        self.patch_feature_extractor = PatchFeatureExtractor(output_embedding_dim=feature_extractor_out_dim)
        self.aggregator = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        aggregation_type=aggregation_type)

        # --- Auxiliary OAI ordinal tasks ---
        # The first one is the most important (KL)
        self.oai_heads = nn.ModuleDict()
        if oai_task_num_classes is not None:
            for task_name, num_cls in oai_task_num_classes.items():
                # Each ordinal head outputs K-1 logits
                self.oai_heads[task_name] = nn.Linear(feature_extractor_out_dim, num_cls - 1)


    def forward(self, list_of_patch_bags):
        batch_logits = []
        batch_att_scores = []
        aggregated_features_all = []
        patch_embeddings_stacked_all = []
        oai_preds_all = {task: [] for task in self.oai_heads}

        for patch_bag_tensor in list_of_patch_bags:
            if patch_bag_tensor.nelement() == 0 or patch_bag_tensor.shape[0] == 0:
                print("Warning: Empty patch bag encountered.")
                for task_name in oai_preds_all:
                    dummy_logit = torch.zeros((1, self.oai_heads[task_name].out_features), device=next(self.parameters()).device)
                    batch_logits.append(dummy_logit)
                    oai_preds_all[task_name].append(dummy_logit.clone())
                continue

            # --- Patch embeddings ---
            patch_embeddings_stacked = self.patch_feature_extractor(patch_bag_tensor)  # (N_i, embedding_dim)
            patch_embeddings_stacked_all.append(patch_embeddings_stacked.cpu())

            # --- MIL aggregation ---
            att_scores, aggregated_features = self.aggregator(patch_embeddings_stacked)
            batch_att_scores.append(att_scores)
            aggregated_features_all.append(aggregated_features.cpu())


            # --- Auxiliary OAI predictions ---
            for task_name, head in self.oai_heads.items():
                oai_pred = head(aggregated_features)  # (1, K-1)
                oai_preds_all[task_name].append(oai_pred)

        # --- Stack results ---
        final_batch_att_scores = torch.cat(batch_att_scores, dim=0)  # (B, num_patches)
        final_aggregated_features = torch.cat(aggregated_features_all, dim=0)  # (B, embedding_dim)
        patch_embeddings_stacked_all = torch.stack(patch_embeddings_stacked_all, dim=0)  # (B, N_i, embedding_dim)

        for task_name in oai_preds_all:
            oai_preds_all[task_name] = torch.cat(oai_preds_all[task_name], dim=0)  # (B, K-1)

        return oai_preds_all, final_batch_att_scores, patch_embeddings_stacked_all, final_aggregated_features, 


    
class CompleteMILOrdinalModel_MultiTask_imedslab(nn.Module):
    def __init__(self, feature_extractor_out_dim, oai_task_num_classes, aggregation_type='attention'):
        super().__init__()
        self.oai_task_num_classes = oai_task_num_classes
        self.patch_feature_extractor = PatchFeatureExtractor(output_embedding_dim=feature_extractor_out_dim)
        self.aggregator = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        aggregation_type=aggregation_type)

        # self.oai_heads = nn.ModuleDict()
        self.oai_heads = Ordinal_MultiTaskHead(n_feats=feature_extractor_out_dim,
                                      n_tasks=(1, len(oai_task_num_classes)-1), 
                                      n_cls=(5, 4),
                                      clf_bnorm=True,
                                      dropout=0.5,
                                      use_gwap=False,
                                      use_gwap_hidden=False,
                                      no_pool=True)
        
        # if oai_task_num_classes is not None:
        #     for task_name, num_cls in oai_task_num_classes.items():
        #         # Each ordinal head outputs K-1 logits
        #         self.oai_heads[task_name] = nn.Linear(feature_extractor_out_dim, num_cls)

    def forward(self, list_of_patch_bags): # list_of_patch_bags: list of tensors, each (N_i, C, H, W)
        batch_logits = []
        batch_att_scores = []
        aggregated_features_all = []
        patch_embeddings_stacked_all = []

        oai_preds_all = {task: [] for task, _ in self.oai_task_num_classes.items()}
        all_task_names = list(self.oai_task_num_classes.keys())

        for patch_bag_tensor in list_of_patch_bags: # Iterate through samples in the batch
            # patch_bag_tensor is (N_i, C, H, W) for the i-th sample in batch
            if patch_bag_tensor.nelement() == 0 or patch_bag_tensor.shape[0] == 0:
                print("Warning: CompleteMILModel encountered an empty patch_bag_tensor for a sample.")
                # Create dummy logits for this problematic sample
                # This ensures the batch processing continues, but this sample won't contribute meaningfully
                dummy_logit = torch.zeros((1, self.aggregator.num_classes), device=next(self.parameters()).device)
                batch_logits.append(dummy_logit)
                continue

            # Process each patch in the bag
            # Need to pass patches one by one or as a batch to feature extractor
            # If feature_extractor takes (B, C, H, W), then patch_bag_tensor is already that.
            patch_embeddings_stacked = self.patch_feature_extractor(patch_bag_tensor) # (N_i, embedding_dim)
            patch_embeddings_stacked_all.append(patch_embeddings_stacked.cpu()) # Store for later use

            # Aggregate and classify
            att_scores, aggregated_features = self.aggregator(patch_embeddings_stacked) # (1, num_classes)
            batch_att_scores.append(att_scores)
            aggregated_features_all.append(aggregated_features.cpu())

             # --- Auxiliary OAI predictions ---
            oai_pred = self.oai_heads(aggregated_features)  # List of logits for each task
            for i in range(len(oai_pred)):
            #     oai_pred = head(aggregated_features)  # (1, K-1)
                # task_name = self.oai_task_num_classes.items()[i][0]
                # print(task_name)
                oai_preds_all[all_task_names[i]].append(oai_pred[i])

                

        final_batch_att_scores = torch.cat(batch_att_scores, dim=0) # (batch_size, num_patches)
        final_aggregated_features = torch.cat(aggregated_features_all, dim=0) # (batch_size, embedding_dim)
        patch_embeddings_stacked_all = torch.stack(patch_embeddings_stacked_all, dim=0) # (batch_size, N_i, embedding_dim)

        for task_name in oai_preds_all:
            oai_preds_all[task_name] = torch.cat(oai_preds_all[task_name], dim=0)  # (B, K-1)

        return oai_preds_all, final_batch_att_scores, patch_embeddings_stacked_all, final_aggregated_features 


class CoralLayer(torch.nn.Module):
    """ Implements CORAL layer described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    -----------
    size_in : int
        Number of input features for the inputs to the forward method, which
        are expected to have shape=(num_examples, num_features).

    num_classes : int
        Number of classes in the dataset.

    preinit_bias : bool (default=True)
        If true, it will pre-initialize the biases to descending values in
        [0, 1] range instead of initializing it to all zeros. This pre-
        initialization scheme results in faster learning and better
        generalization performance in practice.


    """
    def __init__(self, size_in, num_classes, preinit_bias=True):
        super().__init__()
        self.size_in, self.size_out = size_in, 1

        self.coral_weights = torch.nn.Linear(self.size_in, 1, bias=False)
        if preinit_bias:
            self.coral_bias = torch.nn.Parameter(
                torch.arange(num_classes - 1, 0, -1).float() / (num_classes-1))
        else:
            self.coral_bias = torch.nn.Parameter(
                torch.zeros(num_classes-1).float())

    def forward(self, x):
        """
        Computes forward pass.

        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.

        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        return self.coral_weights(x) + self.coral_bias
    
class CompleteMILCoral_MultiTask_Model(nn.Module):
    def __init__(self, feature_extractor_out_dim, oai_task_num_classes, aggregation_type='attention'):
        super().__init__()
        self.patch_feature_extractor = PatchFeatureExtractor(output_embedding_dim=feature_extractor_out_dim)
        self.aggregator = MILAggregator(input_embedding_dim=feature_extractor_out_dim,
                                        aggregation_type=aggregation_type)

        # --- Auxiliary OAI ordinal tasks ---
        # The first one is the most important (KL)
        self.oai_heads = nn.ModuleDict()
        if oai_task_num_classes is not None:
            for task_name, num_cls in oai_task_num_classes.items():
                # Each ordinal head outputs K-1 logits
                self.oai_heads[task_name] = CoralLayer(
                    size_in=feature_extractor_out_dim, 
                    num_classes=num_cls, 
                    preinit_bias=True
                )


    def forward(self, list_of_patch_bags):
        batch_logits = []
        batch_att_scores = []
        aggregated_features_all = []
        patch_embeddings_stacked_all = []
        oai_preds_all = {task: [] for task in self.oai_heads}

        for patch_bag_tensor in list_of_patch_bags:
            if patch_bag_tensor.nelement() == 0 or patch_bag_tensor.shape[0] == 0:
                print("Warning: Empty patch bag encountered.")
                for task_name in oai_preds_all:
                    dummy_logit = torch.zeros((1, self.oai_heads[task_name].out_features), device=next(self.parameters()).device)
                    batch_logits.append(dummy_logit)
                    oai_preds_all[task_name].append(dummy_logit.clone())
                continue

            # --- Patch embeddings ---
            patch_embeddings_stacked = self.patch_feature_extractor(patch_bag_tensor)  # (N_i, embedding_dim)
            patch_embeddings_stacked_all.append(patch_embeddings_stacked.cpu())

            # --- MIL aggregation ---
            att_scores, aggregated_features = self.aggregator(patch_embeddings_stacked)
            batch_att_scores.append(att_scores)
            aggregated_features_all.append(aggregated_features.cpu())


            # --- Auxiliary OAI predictions ---
            for task_name, head in self.oai_heads.items():
                oai_pred = head(aggregated_features)  # (1, K-1)
                oai_preds_all[task_name].append(oai_pred)

        # --- Stack results ---
        final_batch_att_scores = torch.cat(batch_att_scores, dim=0)  # (B, num_patches)
        final_aggregated_features = torch.cat(aggregated_features_all, dim=0)  # (B, embedding_dim)
        patch_embeddings_stacked_all = torch.stack(patch_embeddings_stacked_all, dim=0)  # (B, N_i, embedding_dim)

        for task_name in oai_preds_all:
            oai_preds_all[task_name] = torch.cat(oai_preds_all[task_name], dim=0)  # (B, K-1)

        return oai_preds_all, final_batch_att_scores, patch_embeddings_stacked_all, final_aggregated_features, 
  
