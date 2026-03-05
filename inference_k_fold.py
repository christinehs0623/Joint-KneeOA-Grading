import os
import shutil
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import build_config
from dataset import KneeMILDataset, mil_collate_fn
from losses import coral_predict, coral_multitask_predict
from myutils import (
    calculate_mean_std,
    build_CAM_attention_tool,
    compute_metrics,
    get_criterion,
    labels_to_levels,
    create_transforms,
    prepare_data,
    get_model,
    get_model_org,
    process_CAM, 
    visualize_attention_on_img,
    normalize_attention_scores,
    visualize_cam_comparisons,
    patchFromPoint,
    process_xray,
    create_redsalpha,
)
from sklearn.model_selection import StratifiedKFold
# from inference import run_epoch, Config

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
        all_preds, all_labels, all_probs = [], [], []
        all_attentions = []
        all_patch_embeddings = []
        all_aggregated_features = []
    else:
        all_preds = {task: [] for task in config.OARSI_TASKS.keys()}
        all_labels = {task: [] for task in config.OARSI_TASKS.keys()}
        all_probs = {task: [] for task in config.OARSI_TASKS.keys()}
        all_attentions = {task: [] for task in config.OARSI_TASKS.keys()}
        all_patch_embeddings = {task: [] for task in config.OARSI_TASKS.keys()}
        all_aggregated_features = {task: [] for task in config.OARSI_TASKS.keys()}
        

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
                outputs, att_scores, patch_embeddings, aggregated_features = model(moved_bags)
            else:
                outputs, att_scores, patch_embeddings, aggregated_features = model(moved_bags, model_org, attention_tool)

            # Target handling
            if config.multitask_type == "off":
                loss = criterion(outputs, labels_batch)
                
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
                    for k, v in targets.items():
                        targets[k] = torch.where(v == -999, torch.tensor(0, device=device), v)
 
                                
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

        # Update running loss
        total_loss += loss.item() * labels_batch.size(0)
        num_processed_samples += labels_batch.size(0)

        # Predictions
        if config.multitask_type == "off":
            if config.predict_criteria == "Coral":
                predicted, probs = coral_predict(outputs)
            elif config.predict_criteria == "Max":
                _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            # all_probs.extend(probs.cpu().numpy())
        else:
            if config.predict_criteria == "Coral_Multitask":
                predicted, probs = coral_multitask_predict(outputs)
                for task in config.OARSI_TASKS.keys():
                    all_preds[task].extend(predicted[task][0].cpu().numpy())
                    all_labels[task].extend(targets[task].cpu().numpy())
                    all_attentions[task].extend(att_scores.detach().cpu().numpy())
                    all_patch_embeddings[task].extend(patch_embeddings.detach().cpu().numpy())
                    all_aggregated_features[task].extend(aggregated_features.detach().cpu().numpy())


            elif config.predict_criteria == "Max_Multitask":
                predicted = {}
                for task, out in outputs.items():
                    _, pred = torch.max(out.data, 1)
                    predicted[task] = pred
                for task in config.OARSI_TASKS.keys():
                    all_preds[task].extend(predicted[task].cpu().numpy())
                    all_labels[task].extend(targets[task].cpu().numpy())
                    all_attentions[task].extend(att_scores.detach().cpu().numpy())
                    all_patch_embeddings[task].extend(patch_embeddings.detach().cpu().numpy())
                    all_aggregated_features[task].extend(aggregated_features.detach().cpu().numpy())


                # all_probs[task].extend(probs[task][0].cpu().numpy())

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_processed_samples if num_processed_samples > 0 else 0
    return avg_loss, all_labels, all_preds, all_probs, all_attentions, all_patch_embeddings, all_aggregated_features

def main(config):

    # ----------------- Setup ----------------- #
    files_to_copy = ["inference.py"]
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

    # remove bad cases
    bad_cases = {"9491446_R"}
    mask = ~np.isin(groups, list(bad_cases))
    groups = groups[mask]
    grades = grades[mask]

    # ----------------- FIXED TEST SPLIT (same as train.py) ----------------- #
    split_seed = 42
    train_val, test, train_val_grades, _ = train_test_split(
        groups, grades, test_size=0.2, stratify=grades, random_state=split_seed
    )
    test_pids = test.tolist()

    print(f"[INFO] Test set size = {len(test_pids)}")

    # K-fold on train_val (same as train.py)
    config.K_FOLDS = 5
    kf = StratifiedKFold(
        n_splits=config.K_FOLDS,
        shuffle=True,
        random_state=split_seed
    )

    # ----------------- Load mean/std (same across folds) ----------------- #
    if os.path.exists(config.MEAN_STD_FILE_PATH):
        mean, std = np.load(config.MEAN_STD_FILE_PATH)
    else:
        # compute ONLY from train_val, NOT test
        mean, std = calculate_mean_std(
            config.H5_FILE, train_val.tolist(),
            config.MEAN_STD_FILE_PATH,
            config.DEFAULT_MAX_PIXEL_VALUE
        )
    print(f"Mean: {mean}, Std: {std}")

    _, val_transform = create_transforms(mean, std)

    # ----------------- Dataset & Loader ----------------- #
    test_ds = KneeMILDataset(config.H5_FILE, test_pids, transform=val_transform)
    test_loader = DataLoader(
        test_ds, config.BATCH_SIZE, shuffle=False,
        collate_fn=mil_collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # ======================================================================
    #                           K-FOLD INFERENCE
    # ======================================================================
    # ----------------- Model ----------------- #
    all_fold_preds = []
    all_fold_probs = []
    all_fold_metrics = []
    metrics = "kappa"
    # metrics = "acc"
    # metrics = "f1"

    metrics_per_task = {}   # record acc/f1/kappa across folds


    for fold in range(config.K_FOLDS):

        print(f"\n========== Inference Fold {fold+1}/{config.K_FOLDS} ==========")

        # ----------------- Model ----------------- #
        model = get_model(config)
        model_org = get_model_org(config)

        ckpt_name = f"best_model_avg_{metrics}_fold{fold+1}.pth"
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"[WARNING] Missing checkpoint: {ckpt_name}, skipping this fold.")
            continue

        print(f"[INFO] Loading checkpoint: {ckpt_name}")
        model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))

        if model_org:
            model_org.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH, map_location=config.DEVICE))

        # ----------------- Criterion ----------------- #
        criterion = get_criterion(config.lossfcn_type, None, config.OARSI_TASKS)

        # ----------------- Run inference ----------------- #
        test_loss, test_labels, test_preds, test_probs, \
        all_attentions, all_patch_embeddings, all_aggregated_features = run_epoch(
            test_loader, model, model_org, criterion,
            optimizer=None, device=config.DEVICE,
            is_training=False, config=config,
            desc=f"[Test Fold {fold+1}]"
        )

        print(f"[Fold {fold+1}] Test Loss = {test_loss:.4f}")

        # ----------------- Compute metrics ----------------- #
        test_metrics = compute_metrics(config.multitask_type, test_labels, test_preds)
        all_fold_metrics.append(test_metrics)
        all_fold_preds.append(test_preds)
        all_fold_probs.append(test_probs)

        # ----------------- Save predictions ----------------- #
        np.savez(
            os.path.join(config.CHECKPOINT_DIR, f"test_pred_fold{fold+1}.npz"),
            id=test_pids,
            prob=test_probs,
            pred=test_preds,
            true_kl=test_labels if config.multitask_type == "off" else test_labels["kl"],
            all_attentions=all_attentions,
            all_patch_embeddings=all_patch_embeddings,
            all_aggregated_features=all_aggregated_features,
        )

        # ----------------- Confusion Matrix & Report ----------------- #
        if config.multitask_type == "off":
            task = "kl"
            labels = test_labels
            preds = test_preds

            acc = test_metrics[task]["acc"]
            f1 = test_metrics[task]["f1"]
            kappa = test_metrics[task]["kappa"]

            print(f"[Fold {fold+1}] KL - Acc={acc:.4f} F1={f1:.4f} Kappa={kappa:.4f}")

            report = classification_report(labels, preds)

            # save report
            with open(os.path.join(config.CHECKPOINT_DIR, f"classification_fold{fold+1}.txt"), "w") as f:
                f.write(report)
                f.write("\n")
                f.write(f"Acc   = {acc:.4f}\n")
                f.write(f"F1    = {f1:.4f}\n")
                f.write(f"Kappa = {kappa:.4f}\n")

            # confusion matrix
            ConfusionMatrixDisplay.from_predictions(labels, preds, normalize="true")
            plt.savefig(os.path.join(config.CHECKPOINT_DIR, f"cm_fold{fold+1}.png"))
            plt.close()
            print(report)

            # store for final stats
            if "kl" not in metrics_per_task:
                metrics_per_task["kl"] = {"acc": [], "f1": [], "kappa": []}
            metrics_per_task["kl"]["acc"].append(acc)
            metrics_per_task["kl"]["f1"].append(f1)
            metrics_per_task["kl"]["kappa"].append(kappa)


        else:
            # MULTI-TASK CASE
            for task in test_labels.keys():
                labels = test_labels[task]
                preds = test_preds[task]

                acc = test_metrics[task]["acc"]
                f1 = test_metrics[task]["f1"]
                kappa = test_metrics[task]["kappa"]

                print(f"[Fold {fold+1}] {task} - Acc={acc:.4f} F1={f1:.4f} Kappa={kappa:.4f}")

                report = classification_report(labels, preds)

                # save report
                with open(os.path.join(config.CHECKPOINT_DIR, f"classification_{task}_fold{fold+1}.txt"), "w") as f:
                    f.write(report)
                    f.write("\n")
                    f.write(f"Acc   = {acc:.4f}\n")
                    f.write(f"F1    = {f1:.4f}\n")
                    f.write(f"Kappa = {kappa:.4f}\n")

                # confusion matrix
                ConfusionMatrixDisplay.from_predictions(labels, preds, normalize="true")
                plt.savefig(os.path.join(config.CHECKPOINT_DIR, f"cm_{task}_fold{fold+1}.png"))
                plt.close()
                print(report)

                # store for final stats
                if task not in metrics_per_task:
                    metrics_per_task[task] = {"acc": [], "f1": [], "kappa": []}
                metrics_per_task[task]["acc"].append(acc)
                metrics_per_task[task]["f1"].append(f1)
                metrics_per_task[task]["kappa"].append(kappa)
            
            # save metrics_per_task after each fold
            np.savez(
                os.path.join(config.CHECKPOINT_DIR, f"metrics_per_task_fold{fold+1}.npz"),
                metrics_per_task=metrics_per_task
            )
        plt.close('all')
    
    print(metrics_per_task)
            
    # ----------------- CROSS-FOLD STATISTICAL SUMMARY ----------------- #
    print("\n===== Cross-fold Statistical Summary =====")

    summary_path = os.path.join(config.CHECKPOINT_DIR, f"metrics_summary_{metrics}.txt")
    with open(summary_path, "w") as f:

        for task, vals in metrics_per_task.items():
            acc_arr = np.array(vals["acc"])
            f1_arr = np.array(vals["f1"])
            kappa_arr = np.array(vals["kappa"])

            acc_mean, acc_std = acc_arr.mean(), acc_arr.std()
            f1_mean, f1_std  = f1_arr.mean(),  f1_arr.std()
            kappa_mean, kappa_std = kappa_arr.mean(), kappa_arr.std()

            print(f"\nTask: {task}")
            print(f"  Acc   = {acc_mean:.4f} ± {acc_std:.4f}")
            print(f"  F1    = {f1_mean:.4f} ± {f1_std:.4f}")
            print(f"  Kappa = {kappa_mean:.4f} ± {kappa_std:.4f}")

            f.write(f"Task: {task}\n")
            f.write(f"  Acc   = {acc_mean:.4f} ± {acc_std:.4f}\n")
            f.write(f"  F1    = {f1_mean:.4f} ± {f1_std:.4f}\n")
            f.write(f"  Kappa = {kappa_mean:.4f} ± {kappa_std:.4f}\n\n")

    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    from config import build_config
    config_dict = build_config()
    cfg = Config(config_dict)
    cfg.DEBUG_MODE = True
    cfg.WANDB = False
    main(cfg)

    
    