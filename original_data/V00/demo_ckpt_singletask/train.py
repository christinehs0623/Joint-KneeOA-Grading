import os
import shutil
import numpy as np
import h5py
import json
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import pprint
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
from config import build_config
from dataset import KneeMILDataset, mil_collate_fn
from losses import coral_predict, coral_multitask_predict
from myutils import (
    calculate_mean_std,
    build_CAM_attention_tool,
    compute_metrics,
    compute_class_weights,
    get_criterion,
    labels_to_levels,
    create_transforms,
    prepare_data,
    get_model,
    get_model_org
)
# from pytorch_balanced_sampler import SamplerFactory


def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

def run_epoch(loader, model, model_org, criterion, optimizer, device, is_training, config, desc=""):
    """
    Run one epoch of training/validation
    config: parsed argparse with flags like config.use_multitask, config.use_ordinal, config.training_type
    """
    model.train() if is_training else model.eval()
    total_loss, num_processed_samples = 0.0, 0

    # Prepare prediction containers
    if config.multitask_type == "off":
        all_preds, all_labels = [], []
    else:
        all_preds = {task: [] for task in config.OARSI_TASKS.keys()}
        all_labels = {task: [] for task in config.OARSI_TASKS.keys()}
        

    # Setup attention tool if applicable
    attention_tool = build_CAM_attention_tool(config.feedback_cam, model_org) if model_org else None
    if model_org:
        model_org.eval()

    progress_bar = tqdm(loader, desc=desc, leave=False)

    for list_of_patch_bags, labels_batch, group_name, list_of_features in progress_bar:
        if not list_of_patch_bags:
            continue

        # Move valid bags + features
        moved_bags, moved_features, valid_indices = [], [], []
        for i, bag in enumerate(list_of_patch_bags):
            if bag.nelement() > 0:
                moved_bags.append(bag.to(device, non_blocking=True))
                moved_features.append(list_of_features[i][0].to(device, non_blocking=True))
                valid_indices.append(i)

        if not moved_bags:
            continue

        labels_batch = labels_batch[valid_indices].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            # Forward pass
            if config.feedback_type == "off":
                outputs, _, _, _ = model(moved_bags)
            else:
                outputs, _, _, _ = model(moved_bags, model_org, attention_tool)


            # Target handling
            if config.multitask_type == "off":
                loss = criterion(outputs, labels_batch)
                loss_dict = {'kl': loss.item()}
                
            else:
                if config.multitask_type == "all":
                    targets = {
                        "kl":   labels_batch,
                        "jsnm": torch.tensor([f[0] for f in moved_features], device=device),
                        "jsnl": torch.tensor([f[1] for f in moved_features], device=device),
                        "osfm": torch.tensor([f[2] for f in moved_features], device=device),
                        "ostm": torch.tensor([f[3] for f in moved_features], device=device),
                        "ostl": torch.tensor([f[4] for f in moved_features], device=device),
                        "osfl": torch.tensor([f[5] for f in moved_features], device=device),
                    }
                    # Replace -999 with 0
                    # for k, v in targets.items():
                    #     targets[k] = torch.where(v == -999, torch.tensor(0, device=device), v)
 
                                
                elif config.multitask_type == "kl_jsn":
                    targets = {
                        "kl":   labels_batch,
                        "jsnm": torch.tensor([f[0] for f in moved_features], device=device),
                        "jsnl": torch.tensor([f[1] for f in moved_features], device=device),
                    }
                
                if config.lossfcn_type == "CoralLoss_MultiTask":
                    targets_levels = {}
                    for k, v in targets.items():
                        num_classes = config.OARSI_TASKS[k]  # your dict of num classes per task
                        targets_levels[k] = labels_to_levels(v, num_classes)
                    loss, loss_dict = criterion(outputs, targets_levels)
                else:
                    loss, loss_dict = criterion(outputs, targets)
            
            # Backward
            if is_training:
                loss.backward()
                optimizer.step()
            # print(loss_dict) # {'kl': 1.8507874011993408, 'jsnm': 1.4280171394348145, 'jsnl': 0.6741040945053101, 'osfm': 1.4683774709701538, 'ostm': 1.3698973655700684, 'ostl': 1.265559434890747, 'osfl': 1.4578583240509033
            if config.WANDB and is_training:
                wandb.log(
                    loss_dict,
                )

        # Update running loss
        # total_loss += loss.item() * labels_batch.size(0)

        total_loss += (loss_dict['kl'] + 1 * (loss_dict.get('jsnm',0) + loss_dict.get('jsnl',0) + loss_dict.get('osfm',0) + loss_dict.get('ostm',0) + loss_dict.get('ostl',0) + loss_dict.get('osfl',0))) * labels_batch.size(0)
        num_processed_samples += labels_batch.size(0)

        # Predictions
        if config.multitask_type == "off":
            if config.predict_criteria == "Coral":
                predicted, probs = coral_predict(outputs)
            elif config.predict_criteria == "Max":
                _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
        else:
            if config.predict_criteria == "Coral_Multitask":
                predicted, probs = coral_multitask_predict(outputs)
                for task in config.OARSI_TASKS.keys():
                    all_preds[task].extend(predicted[task][0].cpu().numpy())
                    all_labels[task].extend(targets[task].cpu().numpy())
            elif config.predict_criteria == "Max_Multitask":
                predicted = {}
                for task, out in outputs.items():
                    _, pred = torch.max(out.data, 1)
                    predicted[task] = pred
                for task in config.OARSI_TASKS.keys():
                    all_preds[task].extend(predicted[task].cpu().numpy())
                    all_labels[task].extend(targets[task].cpu().numpy())
            elif config.predict_criteria == "ordinal":
                from losses import ordinal_probs
                for task, out in outputs.items():
                    probs = ordinal_probs(out)           # Tensor
                    preds = torch.argmax(probs, dim=1)   # Tensor
                    all_preds[task].extend(preds.cpu().numpy())
                    all_labels[task].extend(targets[task].cpu().numpy())
                

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_processed_samples if num_processed_samples > 0 else 0
    return avg_loss, all_labels, all_preds, num_processed_samples


    
def save_checkpoint(model, checkpoint_dir, name):
    path = os.path.join(checkpoint_dir, name)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")


    
def log_metrics(metrics_dict, prefix, epoch, use_wandb=True):
    """
    metrics_dict can be multi-task (dict of dicts) or single dict
    """
    if isinstance(metrics_dict[list(metrics_dict.keys())[0]], dict):
        # multi-task
        for task, m in metrics_dict.items():
            line = f"[{prefix}] {task} - Acc: {m['acc']:.4f}, F1: {m['f1']:.4f}, Kappa: {m['kappa']:.4f}"
            print(line)
            if use_wandb:
                wandb.log({
                    f"{prefix}/{task}_accuracy": m['acc'],
                    f"{prefix}/{task}_f1_weighted": m['f1'],
                    f"{prefix}/{task}_kappa": m['kappa'],
                    "epoch": epoch
                })
        # aggregate across tasks
        agg = {
            'acc': np.mean([m['acc'] for m in metrics_dict.values()]),
            'f1': np.mean([m['f1'] for m in metrics_dict.values()]),
            'kappa': np.mean([m['kappa'] for m in metrics_dict.values()])
        }
    else:
        # single task
        m = metrics_dict
        line = f"[{prefix}] Acc: {m['acc']:.4f}, F1: {m['f1']:.4f}, Kappa: {m['kappa']:.4f}"
        print(line)
        if use_wandb:
            wandb.log({
                f"{prefix}/accuracy": m['acc'],
                f"{prefix}/f1_weighted": m['f1'],
                f"{prefix}/kappa": m['kappa'],
                "epoch": epoch
            })
        agg = m
    return agg




def save_best_models(model, metrics_dict, best_metrics, best_mean_metrics, checkpoint_dir):
    """
    metrics_dict: single or multi-task metrics
    best_metrics: dict of best values
    Updates best_metrics and saves checkpoint if new best
    """
    if isinstance(metrics_dict, dict):
        # Multi-task
        kl_metrics = metrics_dict.get("kl", {})
        avg_metrics = {key: np.mean([m[key] for m in metrics_dict.values()]) for key in ['acc','f1','kappa']}

        for key in ['acc','f1','kappa']:
            # KL-best
            kl_val = kl_metrics.get(key, -np.inf)
            if kl_val > best_metrics.get(f"kl_{key}", -np.inf):
                best_metrics[f"kl_{key}"] = kl_val
                path = os.path.join(checkpoint_dir, f"best_model_kl_{key}.pth")
                torch.save(model.state_dict(), path)
                print(f"  Saved new best KL {key} model ({key}: {kl_val:.4f})")

            # AVG-best
            avg_val = avg_metrics[key]
            if avg_val > best_mean_metrics.get(f"avg_{key}", -np.inf):
                best_mean_metrics[f"avg_{key}"] = avg_val
                path = os.path.join(checkpoint_dir, f"best_model_avg_{key}.pth")
                torch.save(model.state_dict(), path)
                print(f"  Saved new best AVG {key} model ({key}: {avg_val:.4f})")

        print("Average metrics across all tasks:")
    else:
        for key in ['acc','f1','kappa']:
            if metrics_dict[key] > best_metrics.get(key, -np.inf):
                best_metrics[key] = metrics_dict[key]
                path = os.path.join(checkpoint_dir, f"best_model_{key}.pth")
                torch.save(model.state_dict(), path)
                print(f"  Saved new best {key} model ({key}: {metrics_dict[key]:.4f})")


def main(config):
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    # ----------------- Setup ----------------- #
    # Copy source files for reproducibility
    files_to_copy = ["train.py", "model.py", "dataset.py", "data_augmentation.py", "losses.py", "utils.py", "config.py"]
    if not config.DEBUG_MODE:
        for file in files_to_copy:
            if os.path.exists(file):
                shutil.copy(file, config.CHECKPOINT_DIR)
                print(f"Copied {file} to {config.CHECKPOINT_DIR}")
            else:
                print(f"WARNING: {file} not found and was not copied.")

        # Initialize wandb
        wandb.init(
            project="Knee_OA_MIL",
            name=config.run_name,
            config=vars(config),
            tags=[
                f"lr{config.LEARNING_RATE:.0e}",
                f"b{config.BATCH_SIZE}",
                f"s{config.SEED}",
                f"e{config.NUM_EPOCHS}"
            ],
        )
    # ----------------- Data ----------------- #
    groups, grades = prepare_data(config.H5_FILE)
    print(f"Total valid samples: {len(groups)}")

    # Train/val/test split
    train_val, test, train_val_grades, _ = train_test_split(
        np.array(groups), np.array(grades), test_size=0.2, stratify=grades, random_state=config.SEED
    )
    train, val, _, _ = train_test_split(
        train_val, train_val_grades, test_size=0.25, stratify=train_val_grades, random_state=config.SEED
    )

    train_pids, val_pids, test_pids = train.tolist(), val.tolist(), test.tolist()
    if "9491446_R" in test_pids:  # remove bad image
        test_pids.remove("9491446_R")

    print(f"Training samples: {len(train_pids)}, Validation: {len(val_pids)}, Testing: {len(test_pids)}")

    # Compute or load mean/std: normalizing input data before feeding it into the model.
    if os.path.exists(config.MEAN_STD_FILE_PATH):
        mean, std = np.load(config.MEAN_STD_FILE_PATH)
    else:
        mean, std = calculate_mean_std(config.H5_FILE, train, config.MEAN_STD_FILE_PATH, config.DEFAULT_MAX_PIXEL_VALUE)
    print(f"Mean: {mean}, Std: {std}")

    train_transform, val_transform = create_transforms(mean, std)

    # Handle DATA_HALF option
    if config.DATA_HALF:
        train_pids, val_pids, test_pids = train_pids[:len(train_pids)//2], val_pids[:len(val_pids)//2], test_pids[:len(test_pids)//2]
        print(f"Using half dataset: train {len(train_pids)}, val {len(val_pids)}, test {len(test_pids)}")

    # ----------------- Class Weight ----------------- #
    # Datasets and loaders
    train_ds = KneeMILDataset(config.H5_FILE, train_pids, transform=train_transform)
    val_ds = KneeMILDataset(config.H5_FILE, val_pids, transform=val_transform)
    test_ds = KneeMILDataset(config.H5_FILE, test_pids, transform=val_transform)
    

    def get_bag_sizes(dataset):
        bag_sizes = []
        for i in range(len(dataset)):
            patches_list, _, group_name, _ = dataset[i]
            bag_sizes.append(len(patches_list))
            if not len(patches_list) == 41:
                print(f"Sample {i} ({group_name}): {len(patches_list)} patches")
        return bag_sizes


    if config.classweight_type == "all_metrics_inv":
        class_weights_tensor_dict = {}
        with h5py.File(config.H5_FILE, 'r') as hf:
            train_kl_grades = []
            jsnm = []
            jsnl = []
            osfm = []
            ostm = []
            ostl = []
            osfl = []
            with h5py.File(config.H5_FILE, 'r') as hf:
                for group_name in train_ds.sample_group_names:
                    train_kl_grades.append(hf[group_name]['kl_grade'][0])
                    jsnm.append(hf[group_name]['aux_feature'][0][0])
                    jsnl.append(hf[group_name]['aux_feature'][0][1])
                    osfm.append(hf[group_name]['aux_feature'][0][2])
                    ostm.append(hf[group_name]['aux_feature'][0][3])
                    ostl.append(hf[group_name]['aux_feature'][0][4])
                    osfl.append(hf[group_name]['aux_feature'][0][5])

                class_counts = np.bincount(train_kl_grades, minlength=config.KL_NUM_CLASSES)
                class_weights_tensor_dict["kl"] = compute_class_weights("inv", class_counts, device=config.DEVICE)
                class_counts_jsnm = np.bincount(jsnm, minlength=4)
                class_weights_tensor_dict["jsnm"] = compute_class_weights("inv", class_counts_jsnm, device=config.DEVICE)
                class_counts_jsnl = np.bincount(jsnl, minlength=4)
                class_weights_tensor_dict["jsnl"] = compute_class_weights("inv", class_counts_jsnl, device=config.DEVICE)

                # class_counts_osfm = np.bincount(osfm, minlength=4)
                # class_weights_tensor_dict["osfm"] = compute_class_weights("inv", class_counts_osfm, device=config.DEVICE)
                # class_counts_ostm = np.bincount(ostm, minlength=4)
                # class_weights_tensor_dict["ostm"] = compute_class_weights("inv", class_counts_ostm, device=config.DEVICE)
                # class_counts_ostl = np.bincount(ostl, minlength=4)
                # class_weights_tensor_dict["ostl"] = compute_class_weights("inv", class_counts_ostl, device=config.DEVICE)
                # class_counts_osfl = np.bincount(osfl, minlength=4)
                # class_weights_tensor_dict["osfl"] = compute_class_weights("inv", class_counts_osfl, device=config.DEVICE)

                # HANDLE -999 LABELS FOR OARSI
                osfm = np.array(osfm)
                ostm = np.array(ostm)
                ostl = np.array(ostl)
                osfl = np.array(osfl)

                class_counts_osfm = np.bincount(np.where(osfm < 0, 0, osfm), minlength=4)
                class_weights_tensor_dict["osfm"] = compute_class_weights("inv", class_counts_osfm, device=config.DEVICE)

                class_counts_ostm = np.bincount(np.where(ostm < 0, 0, ostm), minlength=4)
                class_weights_tensor_dict["ostm"] = compute_class_weights("inv", class_counts_ostm, device=config.DEVICE)

                class_counts_ostl = np.bincount(np.where(ostl < 0, 0, ostl), minlength=4)
                class_weights_tensor_dict["ostl"] = compute_class_weights("inv", class_counts_ostl, device=config.DEVICE)

                class_counts_osfl = np.bincount(np.where(osfl < 0, 0, osfl), minlength=4)
                class_weights_tensor_dict["osfl"] = compute_class_weights("inv", class_counts_osfl, device=config.DEVICE)
                for k, v in class_weights_tensor_dict.items():
                    print(f"Class weights for {k}: {v}")

                
                # IGNORE MISSING LABELS -999
                # class_counts_osfm = np.bincount(np.array(osfm)[np.array(osfm) >= 0], minlength=4)
                # class_weights_tensor_dict["osfm"] = compute_class_weights("inv", class_counts_osfm, device=config.DEVICE)

                # class_counts_ostm = np.bincount(np.array(ostm)[np.array(ostm) >= 0], minlength=4)
                # class_weights_tensor_dict["ostm"] = compute_class_weights("inv", class_counts_ostm, device=config.DEVICE)

                # class_counts_ostl = np.bincount(np.array(ostl)[np.array(ostl) >= 0], minlength=4)
                # class_weights_tensor_dict["ostl"] = compute_class_weights("inv", class_counts_ostl, device=config.DEVICE)

                # class_counts_osfl = np.bincount(np.array(osfl)[np.array(osfl) >= 0], minlength=4)
                # class_weights_tensor_dict["osfl"] = compute_class_weights("inv", class_counts_osfl, device=config.DEVICE)
            
        class_weights_tensor = class_weights_tensor_dict
    else:
        train_kl_grades = []
        with h5py.File(config.H5_FILE, 'r') as hf:
            for group_name in train_ds.sample_group_names:
                train_kl_grades.append(hf[group_name]['kl_grade'][0])
        class_counts = np.bincount(train_kl_grades, minlength=config.KL_NUM_CLASSES)
        print(f"Class counts in training set: {class_counts}")

        class_weights_tensor = compute_class_weights(config.classweight_type, class_counts, device=config.DEVICE)
        print(f"Using class weights: {class_weights_tensor}")


    # if config.balance_sampling:
    #     # sampler =  WeightedRandomSampler(weights = class_weights_tensor, num_samples = len(train_pids), replacement = True)
    #     sampler = SamplerFactory().get(
    #         class_idxs=[0,1,2,3,4],
    #         batch_size=16,
    #         n_batches=336,
    #         alpha=0.5,
    #         kind='fixed'
    #     )
    # else:
    #     sampler = None

    sampler = None

    train_loader = DataLoader(train_ds, config.BATCH_SIZE, shuffle=False if sampler else True, collate_fn=mil_collate_fn, sampler=sampler,
                              num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_ds, config.BATCH_SIZE, shuffle=False, collate_fn=mil_collate_fn,
                            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    test_loader = DataLoader(test_ds, config.BATCH_SIZE, shuffle=False, collate_fn=mil_collate_fn,
                             num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    # ----------------- Model ----------------- #
    model = get_model(config)
    model_org = get_model_org(config)

    # ----------------- Loss & Optimizer ----------------- #
    criterion = get_criterion(config.lossfcn_type, class_weights_tensor, config.OARSI_TASKS)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # ----------------- Training Loop ----------------- #
    # best_metrics = {'kl_acc':0, 'kl_f1':0, 'kl_kappa':-1} 
    best_metrics = {}
    for task in config.OARSI_TASKS.keys():
        best_metrics[f'{task}_acc'] = 0
        best_metrics[f'{task}_f1'] = 0
        best_metrics[f'{task}_kappa'] = -1

    best_mean_metrics = {'avg_acc':0, 'avg_f1':0, 'avg_kappa':-1}

    for epoch in range(config.NUM_EPOCHS):
        torch.cuda.empty_cache()
        epoch_num = epoch + 1

        # Train
        train_loss, train_labels, train_preds, processed_train_samples = run_epoch(
            train_loader, model, model_org, criterion, optimizer, config.DEVICE,
            is_training=True, config=config,
            desc=f"Epoch {epoch_num}/{config.NUM_EPOCHS} [Train]"
        )
        if processed_train_samples > 0:
            train_metrics = compute_metrics(config.multitask_type, train_labels, train_preds)
            log_metrics(train_metrics, "Train", epoch_num, use_wandb=config.WANDB)

        # Validate
        val_loss, val_labels, val_preds, processed_val_samples = run_epoch(
            val_loader, model, model_org, criterion, None, config.DEVICE,
            is_training=False, config=config,
            desc=f"Epoch {epoch_num}/{config.NUM_EPOCHS} [Val]"
        )
        scheduler.step(val_loss)

        if processed_val_samples > 0:
            val_metrics = compute_metrics(config.multitask_type, val_labels, val_preds)
            log_metrics(val_metrics, "Val", epoch_num, use_wandb=config.WANDB)

        save_best_models(model, val_metrics, best_metrics, best_mean_metrics, config.CHECKPOINT_DIR)

    print("Training finished.")
    if config.WANDB:
        wandb.finish()


if __name__ == "__main__":
    from config import build_config
    config_dict = build_config()

    DEVICE = config_dict["DEVICE"]
    config_dict["DEVICE"] = str(DEVICE)
    config_path = os.path.join(config_dict["CHECKPOINT_DIR"], "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"Config saved to: {config_path}")

    config_dict["DEVICE"] = DEVICE
    cfg = Config(config_dict)
    main(cfg)
