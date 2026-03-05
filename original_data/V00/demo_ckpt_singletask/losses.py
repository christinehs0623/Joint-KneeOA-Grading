from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum

def normalize_weights(w: torch.Tensor):
    return w / w.mean()

# class CrossEntropy_MultiTask(nn.Module):
#     def __init__(self, class_weights=None):
#         super(CrossEntropy_MultiTask, self).__init__()
#         self.class_weights = class_weights

#     def forward(self, outputs, targets):
#         """
#         Args:
#             outputs logits: (batch_size, num_classes)
#             targets: (batch_size,)
#         """
#         total_loss = 0
#         loss_dict = {}
        
#         for task, preds in outputs.items():
#             if task not in targets:
#                 continue

#             l = F.cross_entropy(outputs[task], targets[task].long(), reduction="mean")
            
#             total_loss += l
#             loss_dict[task] = l.item()

#         total_loss = total_loss / len(outputs)  # average over tasks

#         return total_loss, loss_dict

# class CrossEntropy_MultiTask(nn.Module):
#     def __init__(self, class_weights=None):
#         super(CrossEntropy_MultiTask, self).__init__()
#         self.class_weights = {}
#         if class_weights is not None:
#             # normalize each task’s weight tensor
#             for task, w in class_weights.items():
#                 if w is not None:
#                     self.class_weights[task] = w / w.mean()
#                 else:
#                     self.class_weights[task] = None

#     def forward(self, outputs, targets):
#         total_loss = 0
#         loss_dict = {}
#         num_tasks = 0
        
#         for task, preds in outputs.items():
#             if task not in targets:
#                 continue

#             weights = self.class_weights.get(task, None)
#             l = F.cross_entropy(
#                 preds, 
#                 targets[task].long(),
#                 weight=weights,
#                 reduction="mean"
#             )
            
#             total_loss += l
#             loss_dict[task] = l.item()
#             num_tasks += 1

            
                
#         if num_tasks > 0:
#             total_loss = total_loss / num_tasks

#         return total_loss, loss_dict

class CrossEntropy_MultiTask(nn.Module):
    def __init__(self, class_weights=None):
        super(CrossEntropy_MultiTask, self).__init__()
        self.class_weights = {}
        if class_weights is not None:
            for task, w in class_weights.items():
                if w is not None:
                    # 確保 weight 在正確的 device 上
                    self.class_weights[task] = w / w.mean()
                else:
                    self.class_weights[task] = None

    def forward(self, outputs, targets):
        total_loss = 0
        loss_dict = {}
        num_tasks = 0

        for task, preds in outputs.items():
            if task not in targets:
                continue

            # 針對每個任務單獨建立 mask
            target = targets[task].long()
            task_mask = (target != -999)

            # 如果這個 Batch 裡該任務全部都是 -999，就跳過
            if not task_mask.any():
                loss_dict[task] = 0.0
                continue

            # 關鍵修正：只挑選有效樣本
            # preds 形狀是 [Batch, Num_Classes]，target 形狀是 [Batch]
            masked_preds = preds[task_mask]     # 形狀變為 [Valid_N, Num_Classes]
            masked_target = target[task_mask]   # 形狀變為 [Valid_N]

            weights = self.class_weights.get(task, None)
            if weights is not None:
                weights = weights.to(preds.device)

            # 計算該任務的平均 Loss
            l = F.cross_entropy(
                masked_preds, 
                masked_target, 
                weight=weights,
                reduction="mean"
            )
            
            # --- 權重平衡建議 ---
            # 如果你想讓 KL 更好，可以手動給 KL 更大的權重
            task_weight = 1.0 # if task == 'kl' else 0.2
            total_loss += task_weight * l
            # ------------------

            loss_dict[task] = l.item()
            num_tasks += 1

        # if num_tasks > 0:
        #     total_loss = total_loss / num_tasks

        return total_loss, loss_dict

class CoralLossWeighted(nn.Module):
    """
    CORAL loss with class weights to handle imbalanced data
    """
    def __init__(self, class_weights=None):
        """
        Args:
            class_weights: Tensor of shape (num_classes,)
                           對應到每個 class 的權重，例如 [w0, w1, w2, w3, w4]
        """
        super(CoralLossWeighted, self).__init__()
        self.class_weights = class_weights

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, K-1)
            targets: (batch_size,)
        """
        batch_size, num_classes_minus1 = logits.shape
        prob = torch.sigmoid(logits)

        # 建立 target matrix (binary)
        target_matrix = torch.zeros((batch_size, num_classes_minus1), device=logits.device)
        
        for i in range(batch_size):
            target_matrix[i, :targets[i]] = 1

        # 如果有 class weight
        if self.class_weights is not None:
            # 依照每個樣本的 true label 取對應權重
            sample_weights = self.class_weights[targets]   # (batch_size,)
            # 擴展到 (batch_size, K-1)，讓每個 cutpoint 都能乘上
            sample_weights = sample_weights.unsqueeze(1).expand_as(prob)
        else:
            sample_weights = torch.ones_like(prob)

        # Binary cross-entropy with weights
        loss = F.binary_cross_entropy(prob, target_matrix, weight=sample_weights, reduction="mean")

        return loss


class CoralLossEffective(nn.Module):
    """
    CORAL loss with per-threshold effective number weighting
    """
    def __init__(self, threshold_weights=None):
        """
        Args:
            threshold_weights: Tensor of shape (K-1, 2),
                               每個 threshold 的 [w_neg, w_pos]
        """
        super(CoralLossEffective, self).__init__()
        self.threshold_weights = threshold_weights  # (K-1, 2)

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, K-1)
            targets: (batch_size,)
        """
        batch_size, num_classes_minus1 = logits.shape
        prob = torch.sigmoid(logits)

        # target_matrix: (batch_size, K-1)\
        target_matrix = torch.zeros((batch_size, num_classes_minus1), device=logits.device)
        for i in range(batch_size):
            target_matrix[i, :targets[i]] = 1

        # per-threshold BCE with weights
        loss_matrix = torch.zeros_like(prob)
        for k in range(num_classes_minus1):
            w_neg, w_pos = self.threshold_weights[k]

            # BCE 分開寫正負
            loss_matrix[:, k] = - (
                w_pos * target_matrix[:, k] * torch.log(prob[:, k] + 1e-8) +
                w_neg * (1 - target_matrix[:, k]) * torch.log(1 - prob[:, k] + 1e-8)
            )

        return loss_matrix.mean()
    
class CoralFocalLoss(nn.Module):
    """
    Focal-CORAL loss with optional class weights for imbalanced data.
    """
    def __init__(self, class_weights=None, gamma=2.0, alpha=0.25):
        """
        Args:
            class_weights: Tensor of shape (num_classes,), optional class-level weight
            gamma: focusing parameter, typical value 2.0
            alpha: balance parameter between positive/negative samples
        """
        super(CoralFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (batch_size, K-1), raw outputs
            targets: Tensor of shape (batch_size,), true labels (0~K-1)
        """
        batch_size, num_classes_minus1 = logits.shape
        prob = torch.sigmoid(logits)

        # 建立 target matrix (binary)
        target_matrix = torch.zeros((batch_size, num_classes_minus1), device=logits.device)
        for i in range(batch_size):
            target_matrix[i, :targets[i]] = 1

        # 計算 p_t (對應正負樣本)
        pt = torch.where(target_matrix == 1, prob, 1 - prob)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(target_matrix == 1, self.alpha, 1 - self.alpha)

        # Focal-CORAL loss matrix
        loss_matrix = - alpha_t * focal_weight * (
            target_matrix * torch.log(prob + 1e-8) +
            (1 - target_matrix) * torch.log(1 - prob + 1e-8)
        )

        # 加上 class weight
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]  # (batch,)
            sample_weights = sample_weights.unsqueeze(1).expand_as(loss_matrix)
            loss_matrix = loss_matrix * sample_weights

        return loss_matrix.mean()

class CoralFocalLoss_MultiTask(nn.Module):

    """
    Focal-CORAL loss with optional class weights for imbalanced data.
    """
    def __init__(self, task_num_classes, is_learn_task_weights, class_weights=None, gamma=2.0, alpha=0.25):
        """
        Args:
            class_weights: Tensor of shape (num_classes,), optional class-level weight
            gamma: focusing parameter, typical value 2.0
            alpha: balance parameter between positive/negative samples
            learn_task_weights: if True, learn uncertainty-based weights for tasks
        """
        super(CoralFocalLoss_MultiTask, self).__init__()
        self.task_num_classes = task_num_classes
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha

        if is_learn_task_weights:
            self.log_vars = nn.ParameterDict({
                t: nn.Parameter(torch.zeros(1)) for t in task_num_classes
            })
        else:
            self.log_vars = None

    def coral_focal_loss(self, logits, targets, kl_logits): # kl_logits is for class weights
        """
        Args:
            logits: Tensor of shape (batch_size, K-1), raw outputs
            targets: Tensor of shape (batch_size,), true labels (0~K-1)
        """
        batch_size, num_classes_minus1 = logits.shape
        prob = torch.sigmoid(logits)
        if self.log_vars:
            self.log_vars.to(logits.device)

        # 建立 target matrix (binary)
        target_matrix = torch.zeros((batch_size, num_classes_minus1), device=logits.device)
        for i in range(batch_size):
            target_matrix[i, :int(targets[i])] = 1

        # 計算 p_t (對應正負樣本)
        pt = torch.where(target_matrix == 1, prob, 1 - prob)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(target_matrix == 1, self.alpha, 1 - self.alpha)

        # Focal-CORAL loss matrix
        loss_matrix = - alpha_t * focal_weight * (
            target_matrix * torch.log(prob + 1e-8) +
            (1 - target_matrix) * torch.log(1 - prob + 1e-8)
        )

        if self.class_weights is not None:
            # sample_weights: [batch] -> [batch, 1] so it can broadcast along num_classes
            sample_weights = self.class_weights[kl_logits].view(-1, 1) 
            loss_matrix = loss_matrix * sample_weights  # broadcast automatically

        return loss_matrix.mean()

    def forward(self, outputs, targets):
        """
        Args:

            outputs:{
                "kl":
                "jsnm":
                "jsnl":
            }

            targets: dict of labels
        """
        total_loss = 0
        loss_dict = {}

        # print("Outputs keys:", outputs.keys())
        for task, preds in outputs.items():
            if task not in targets:
                continue

            l = self.coral_focal_loss(
                preds,
                targets[task],
                kl_logits=targets["kl"]
            )
            
            total_loss += l
            loss_dict[task] = l.item()

        total_loss = total_loss / len(outputs)  # average over tasks

        return total_loss, loss_dict

def coral_predict(logits):
    """
    logits: (batch_size, K-1)
    return: predicted class (batch_size,)
    """
    prob = torch.sigmoid(logits)   # (batch, K-1)
    preds = torch.sum(prob > 0.5, dim=1)
    return preds, prob

def coral_multitask_predict(outputs):
    """
    outputs: dict of task_name -> logits
    returns: dict of task_name -> predicted labels
    """
    preds = {}
    probs = {}
    for task, logits in outputs.items():
        preds[task] = coral_predict(logits)
        probs[task] = coral_predict(logits)
    return preds, probs


# class CoralLoss_MultiTask(torch.nn.Module):
#     """Computes the CORAL loss described in

#     Cao, Mirjalili, and Raschka (2020)
#     *Rank Consistent Ordinal Regression for Neural Networks
#        with Application to Age Estimation*
#     Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

#     Parameters
#     ----------
#     reduction : str or None (default='mean')
#         If 'mean' or 'sum', returns the averaged or summed loss value across
#         all data points (rows) in logits. If None, returns a vector of
#         shape (num_examples,)

#     Examples
#     ----------
#     >>> import torch
#     >>> from coral_pytorch.losses import CoralLoss
#     >>> levels = torch.tensor(
#     ...    [[1., 1., 0., 0.],
#     ...     [1., 0., 0., 0.],
#     ...    [1., 1., 1., 1.]])
#     >>> logits = torch.tensor(
#     ...    [[2.1, 1.8, -2.1, -1.8],
#     ...     [1.9, -1., -1.5, -1.3],
#     ...     [1.9, 1.8, 1.7, 1.6]])
#     >>> loss = CoralLoss()
#     >>> loss(logits, levels)
#     tensor(0.6920)
#     """

#     def __init__(self, reduction='mean', class_weights=None):
#         super().__init__()
#         self.reduction = reduction
#         self.class_weights= class_weights

#     def coral_loss(self, logits, levels, kl_logits, importance_weights=None, reduction='mean'):
#         """Computes the CORAL loss described in

#         Cao, Mirjalili, and Raschka (2020)
#         *Rank Consistent Ordinal Regression for Neural Networks
#         with Application to Age Estimation*
#         Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

#         Parameters
#         ----------
#         logits : torch.tensor, shape(num_examples, num_classes-1)
#             Outputs of the CORAL layer.

#         levels : torch.tensor, shape(num_examples, num_classes-1)
#             True labels represented as extended binary vectors
#             (via `coral_pytorch.dataset.levels_from_labelbatch`).

#         importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
#             Optional weights for the different labels in levels.
#             A tensor of ones, i.e.,
#             `torch.ones(num_classes-1, dtype=torch.float32)`
#             will result in uniform weights that have the same effect as None.

#         reduction : str or None (default='mean')
#             If 'mean' or 'sum', returns the averaged or summed loss value across
#             all data points (rows) in logits. If None, returns a vector of
#             shape (num_examples,)

#         Returns
#         ----------
#             loss : torch.tensor
#             A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
#             or a loss value for each data record (if `reduction=None`).

#         Examples
#         ----------
#         >>> import torch
#         >>> from coral_pytorch.losses import coral_loss
#         >>> levels = torch.tensor(
#         ...    [[1., 1., 0., 0.],
#         ...     [1., 0., 0., 0.],
#         ...    [1., 1., 1., 1.]])
#         >>> logits = torch.tensor(
#         ...    [[2.1, 1.8, -2.1, -1.8],
#         ...     [1.9, -1., -1.5, -1.3],
#         ...     [1.9, 1.8, 1.7, 1.6]])
#         >>> coral_loss(logits, levels)
#         tensor(0.6920)
#         """

#         if not logits.shape == levels.shape:
#             raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
#                             % (logits.shape, levels.shape))

#         term1 = (F.logsigmoid(logits)*levels
#                         + (F.logsigmoid(logits) - logits)*(1-levels))

#         if importance_weights is not None:
#             sample_weights = importance_weights[kl_logits].view(-1, 1) 
#             term1 *= sample_weights  # broadcast automatically

#         val = (-torch.sum(term1, dim=1))

#         if reduction == 'mean':
#             loss = torch.mean(val)
#         elif reduction == 'sum':
#             loss = torch.sum(val)
#         elif reduction is None:
#             loss = val
#         else:
#             s = ('Invalid value for `reduction`. Should be "mean", '
#                 '"sum", or None. Got %s' % reduction)
#             raise ValueError(s)

#         return loss
    
#     # def forward(self, logits, levels, importance_weights=None):
#     #     """
#     #     Parameters
#     #     ----------
#     #     logits : torch.tensor, shape(num_examples, num_classes-1)
#     #         Outputs of the CORAL layer.

#     #     levels : torch.tensor, shape(num_examples, num_classes-1)
#     #         True labels represented as extended binary vectors
#     #         (via `coral_pytorch.dataset.levels_from_labelbatch`).

#     #     importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
#     #         Optional weights for the different labels in levels.
#     #         A tensor of ones, i.e.,
#     #         `torch.ones(num_classes-1, dtype=torch.float32)`
#     #         will result in uniform weights that have the same effect as None.
#     #     """
#     #     return self.coral_loss(
#     #         logits, levels,
#     #         importance_weights=importance_weights,
#     #         reduction=self.reduction)
    
#     def forward(self, outputs, targets):
#         """
#         Args:

#             outputs:{
#                 "kl":
#                 "jsnm":
#                 "jsnl":
#             }

#             targets: dict of labels
#         """
#         total_loss = 0
#         loss_dict = {}

#         # print("Outputs keys:", outputs.keys())
#         for task, logits in outputs.items():
#             if task not in targets:
#                 continue

#             l = self.coral_loss(
#                 logits, targets[task], targets["kl"].sum(dim=1).long(),
#                 importance_weights=self.class_weights,
#                 reduction=self.reduction)

#             total_loss += l
#             loss_dict[task] = l.item()

#         total_loss = total_loss / len(outputs)  # average over tasks

#         return total_loss, loss_dict
    

class CoralFocalLoss_MultiTask_MetricsBalanced(nn.Module):
    """
    Multi-task Focal-CORAL loss with task-specific class weights and optional learnable task weights.
    """
    def __init__(self, task_num_classes, is_learn_task_weights=False, class_weights=None, gamma=2.0, alpha=0.25):
        """
        Args:
            task_num_classes: dict, {task_name: num_classes}
            is_learn_task_weights: if True, learn uncertainty-based weights per task
            class_weights: dict of {task_name: Tensor(num_classes,)} or None
            gamma: focusing parameter for focal loss
            alpha: balance parameter between positive/negative samples
        """
        super().__init__()
        self.task_num_classes = task_num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights

        if is_learn_task_weights:
            self.log_vars = nn.ParameterDict({
                t: nn.Parameter(torch.zeros(1)) for t in task_num_classes
            })
        else:
            self.log_vars = None

    def coral_focal_loss(self, logits, targets, class_weights=None):
        """
        Args:
            logits: Tensor(batch_size, K-1)
            targets: Tensor(batch_size,)
            class_weights: Tensor(K,) or None
        """
        batch_size, num_classes_minus1 = logits.shape
        prob = torch.sigmoid(logits)

        # target matrix
        target_matrix = torch.zeros((batch_size, num_classes_minus1), device=logits.device)
        for i in range(batch_size):
            target_matrix[i, :int(targets[i])] = 1

        # focal terms
        pt = torch.where(target_matrix == 1, prob, 1 - prob)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(target_matrix == 1, self.alpha, 1 - self.alpha)

        loss_matrix = - alpha_t * focal_weight * (
            target_matrix * torch.log(prob + 1e-8) +
            (1 - target_matrix) * torch.log(1 - prob + 1e-8)
        )

        # class weights
        if class_weights is not None:
            # expand weights to binary comparisons (K-1 per sample)
            sample_weights = class_weights[targets.long()].view(-1, 1)
            loss_matrix = loss_matrix * sample_weights

        return loss_matrix.mean()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict {task_name: logits(batch, K-1)}
            targets: dict {task_name: labels(batch,)}
        """
        total_loss = 0
        loss_dict = {}

        for task, preds in outputs.items():
            if task not in targets:
                continue
            # print(self.class_weights)
            if self.class_weights is None:
                class_w = None
            else:   
                class_w = self.class_weights.get(task, None)

            l = self.coral_focal_loss(preds, targets[task], class_w)

            if self.log_vars is not None:  # uncertainty weighting
                self.log_vars.to(l.device)
                precision = torch.exp(-self.log_vars[task])
                l = precision * l + self.log_vars[task]

            total_loss += l
            loss_dict[task] = l.item()

        total_loss = total_loss / len(loss_dict)  # normalize by number of valid tasks

        return total_loss, loss_dict

def cumulative_target(targets, num_classes):
    """
    targets: (B,) int labels
    num_classes: K
    return: (B, K-1) cumulative encoding
    """
    B = targets.size(0)
    out = torch.zeros(B, num_classes, device=targets.device)
    for k in range(0, num_classes):
        out[:, k] = (targets >= k).float()
    return out

class BCEWithLogitsLoss_MultiTask(nn.Module):
    def __init__(self, class_weights=None):
        super(BCEWithLogitsLoss_MultiTask, self).__init__()
        self.class_weights = class_weights

    def forward(self, outputs, targets):
        """
        Args:
            outputs logits: (batch_size, num_classes)
            targets: (batch_size,)
        """
        total_loss = 0
        loss_dict = {}
        
        
        for task, preds in outputs.items():
            if task not in targets:
                continue
            
            target_cum = cumulative_target(targets[task], preds.size(1)) # should make targets cumulative
            # class_counts: 原始 K 維
            if self.class_weights is None:
                pos_weight_tensor = None
            else:
                class_counts = self.class_weights[task]  # 例如 tensor([c0, c1, c2, c3, c4])
                K = len(class_counts)

                pos_weight = []
                total = class_counts.sum().item()
                for k in range(1, K):
                    pos = class_counts[k:].sum().item()   # y >= k
                    neg = total - pos                     # y < k
                    pos_weight.append(neg / pos)

                pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(preds.device)
            
            # print(preds.shape, target_cum.shape, pos_weight_tensor.shape)
            l = F.binary_cross_entropy_with_logits(
                preds, 
                target_cum, 
                pos_weight=pos_weight_tensor,
                reduction="mean"
            )
            
            total_loss += l
            loss_dict[task] = l.item()

        total_loss = total_loss / len(outputs)  # average over tasks

        return total_loss, loss_dict

def ordinal_probs(prob):
    # prob: (B, K-1)
    B, K_minus_1 = prob.shape
    prob_full = torch.zeros((B, K_minus_1 + 1), device=prob.device)

    prob_full[:, 0] = 1 - prob[:, 0]
    for k in range(1, K_minus_1):
        prob_full[:, k] = prob[:, k-1] - prob[:, k]
    prob_full[:, -1] = prob[:, -1]

    return prob_full  # (B, K)


class CoralLoss_MultiTask(torch.nn.Module):
    """Computes the CORAL loss described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    ----------
    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)

    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import CoralLoss
    >>> levels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> loss = CoralLoss()
    >>> loss(logits, levels)
    tensor(0.6920)
    """

    def __init__(self, reduction='mean', class_weights=None):
        super().__init__()
        self.reduction = reduction
        self.class_weights= class_weights

    def coral_loss(self, logits, levels, importance_weights=None, reduction='mean'):
        """Computes the CORAL loss described in

        Cao, Mirjalili, and Raschka (2020)
        *Rank Consistent Ordinal Regression for Neural Networks
        with Application to Age Estimation*
        Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

        Parameters
        ----------
        logits : torch.tensor, shape(num_examples, num_classes-1)
            Outputs of the CORAL layer.

        levels : torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
            (via `coral_pytorch.dataset.levels_from_labelbatch`).

        importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
            Optional weights for the different labels in levels.
            A tensor of ones, i.e.,
            `torch.ones(num_classes-1, dtype=torch.float32)`
            will result in uniform weights that have the same effect as None.

        reduction : str or None (default='mean')
            If 'mean' or 'sum', returns the averaged or summed loss value across
            all data points (rows) in logits. If None, returns a vector of
            shape (num_examples,)

        Returns
        ----------
            loss : torch.tensor
            A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
            or a loss value for each data record (if `reduction=None`).

        Examples
        ----------
        >>> import torch
        >>> from coral_pytorch.losses import coral_loss
        >>> levels = torch.tensor(
        ...    [[1., 1., 0., 0.],
        ...     [1., 0., 0., 0.],
        ...    [1., 1., 1., 1.]])
        >>> logits = torch.tensor(
        ...    [[2.1, 1.8, -2.1, -1.8],
        ...     [1.9, -1., -1.5, -1.3],
        ...     [1.9, 1.8, 1.7, 1.6]])
        >>> coral_loss(logits, levels)
        tensor(0.6920)
        """

        if not logits.shape == levels.shape:
            raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                            % (logits.shape, levels.shape))

        term1 = (F.logsigmoid(logits)*levels
                        + (F.logsigmoid(logits) - logits)*(1-levels))

        if importance_weights is not None:
            sample_weights = torch.tensor(
                importance_weights,
                dtype=logits.dtype,
                device=logits.device
            ).view(1, -1)
            term1 = term1 * sample_weights

        val = (-torch.sum(term1, dim=1))

        if reduction == 'mean':
            loss = torch.mean(val)
        elif reduction == 'sum':
            loss = torch.sum(val)
        elif reduction is None:
            loss = val
        else:
            s = ('Invalid value for `reduction`. Should be "mean", '
                '"sum", or None. Got %s' % reduction)
            raise ValueError(s)

        return loss
    
    def forward(self, outputs, targets):
        """
        Args:

            outputs:{
                "kl":
                "jsnm":
                "jsnl":
            }

            targets: dict of labels
        """
        total_loss = 0
        loss_dict = {}

        # print("Outputs keys:", outputs.keys())
        for task, logits in outputs.items():
            if task not in targets:
                continue
            
            # accumulated class weights for CORAL
            pos_weight = None
            if self.class_weights is not None:
                weights = self.class_weights.get(task, None)
                class_counts = weights  # 例如 tensor([c0, c1, c2, c3, c4])
                K = len(class_counts) # num_classes
                pos_weight = []
                total = class_counts.sum().item()
                for k in range(1, K):
                    pos = class_counts[k:].sum().item()   # y >= k
                    pos_weight.append(total / pos)
                    # neg = total - pos                     # y < k
                    # pos_weight.append(neg / pos)

            l = self.coral_loss(
                logits, targets[task],
                importance_weights=pos_weight,
                reduction=self.reduction)

            total_loss += l
            loss_dict[task] = l

        total_loss = total_loss / len(outputs)  # average over tasks

        return total_loss, loss_dict