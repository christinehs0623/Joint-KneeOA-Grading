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
from sklearn.model_selection import StratifiedKFold
from train import Config, run_epoch, grad_norm, log_metrics



def save_best_models(model, metrics_dict, best_metrics, best_mean_metrics, checkpoint_dir, fold=None):
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
                if fold is not None:
                    path = os.path.join(checkpoint_dir, f"best_model_avg_{key}_fold{fold+1}.pth")
                else:
                    path = os.path.join(checkpoint_dir, f"best_model_avg_{key}.pth")
                torch.save(model.state_dict(), path)
                print(f"  Saved new best AVG {key} model ({key}: {avg_val:.4f})")

        print("Average metrics across all tasks:")
    else:
        for key in ['acc','f1','kappa']:
            if metrics_dict[key] > best_metrics.get(key, -np.inf):
                best_metrics[key] = metrics_dict[key]
                if fold is not None:
                    path = os.path.join(checkpoint_dir, f"best_model_{key}_fold{fold+1}.pth")
                else:
                    path = os.path.join(checkpoint_dir, f"best_model_{key}.pth")
                torch.save(model.state_dict(), path)
                print(f"  Saved new best {key} model ({key}: {metrics_dict[key]:.4f})")


def main(config):
    split_seed = 42
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    # ----------------- Setup ----------------- #
    # Copy source files for reproducibility
    files_to_copy = ["train_k_fold.py", "model.py", "dataset.py", "data_augmentation.py", "losses.py", "utils.py", "config.py"]
    if not config.DEBUG_MODE:
        for file in files_to_copy:
            if os.path.exists(file):
                shutil.copy(file, config.CHECKPOINT_DIR)
                print(f"Copied {file} to {config.CHECKPOINT_DIR}")
            else:
                print(f"WARNING: {file} not found and was not copied.")

    # ----------------- Data ----------------- #
    groups, grades = prepare_data(config.H5_FILE)
    print(f"Total valid samples: {len(groups)}")

    groups = np.array(groups)
    grades = np.array(grades)
    bad_cases = {"9491446_R"} # only have 33 patches instead of 41
    mask = ~np.isin(groups, list(bad_cases))

    # apply mask
    groups = groups[mask]
    grades = grades[mask]

    # Train/val/test split (K fold: test is fixed.)
    train_val, test, train_val_grades, _ = train_test_split(
        groups, grades, test_size=0.2, stratify=grades, random_state=split_seed
    )
    test_pids = test.tolist()

    config.K_FOLDS = 5
    kf = StratifiedKFold(
        n_splits=config.K_FOLDS,
        shuffle=True,
        random_state=split_seed
    )
    fold_results = []

    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val, train_val_grades)):
        if config.WANDB:
            wandb.init(
                project="Knee_OA_MIL",
                name=config.run_name + f"_fold{fold+1}",
                config=vars(config),
                tags=[
                    f"lr{config.LEARNING_RATE:.0e}",
                    f"b{config.BATCH_SIZE}",
                    f"s{config.SEED}",
                    f"e{config.NUM_EPOCHS}"
                ],
            )
        print(f"\n======= Fold {fold+1}/{config.K_FOLDS} =======")
        train_pids = groups[train_idx].tolist()
        val_pids   = groups[val_idx].tolist()

        # Compute or load mean/std: normalizing input data before feeding it into the model.
        mean, std = calculate_mean_std(config.H5_FILE, train_pids, config.MEAN_STD_FILE_PATH, config.DEFAULT_MAX_PIXEL_VALUE)
        print(f"Mean: {mean}, Std: {std}")
        print(f"Training samples: {len(train_pids)}, Validation: {len(val_pids)}, Testing: {len(test_pids)}")

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

            save_best_models(model, val_metrics, best_metrics, best_mean_metrics, config.CHECKPOINT_DIR, fold)

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
