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
    visualize_raw_xray_only,
    visualize_dicom_only,
    visualize_attention_on_img,
    normalize_attention_scores,
    visualize_cam_comparisons,
    patchFromPoint,
    process_xray,
    create_redsalpha,
)
from sklearn.utils import resample

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
        all_preds, all_labels, all_probs,  all_attentions, all_patch_embeddings, all_aggregated_features  = [], [], [], [], [], []
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
                moved_bags.append(bag.to(device, non_blocking=config.PIN_MEMORY))
                moved_features.append(list_of_features[i][0].to(device, non_blocking=config.PIN_MEMORY))
                valid_indices.append(i)

        if not moved_bags:
            continue

        labels_batch = labels_batch[valid_indices].to(device, non_blocking=config.PIN_MEMORY)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            # Forward pass
            if config.feedback_type == "off":
                if config.model_type == "MIL_ORG":
                    outputs, _ = model(moved_bags)
                    # print(outputs)
                    
                else:
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

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_processed_samples if num_processed_samples > 0 else 0
    return avg_loss, all_labels, all_preds, all_probs, all_attentions, all_patch_embeddings, all_aggregated_features

def bootstrap_evaluation(
    model, model_org, test_pids, config, criterion, n_iterations=5
):
    all_iteration_metrics = []

    print("\n" + "=" * 40)
    print(f"🌟 Starting Bootstrapping (N={n_iterations})")
    print("=" * 40)

    for i in range(n_iterations):
        bs_pids = resample(
            test_pids,
            replace=True,
            n_samples=len(test_pids),
            random_state=config.SEED + i
        )

        bs_ds = KneeMILDataset(
            config.H5_FILE,
            bs_pids,
            transform=create_transforms(*np.load(config.MEAN_STD_FILE_PATH))[1]
        )

        bs_loader = DataLoader(
            bs_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=mil_collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        _, bs_labels, bs_preds, _, _, _, _ = run_epoch(
            bs_loader,
            model,
            model_org,
            criterion,
            optimizer=None,
            device=config.DEVICE,
            is_training=False,
            config=config,
            desc=f"BS-Iter {i+1}"
        )

        metrics = compute_metrics(
            config.multitask_type, bs_labels, bs_preds
        )

        all_iteration_metrics.append(metrics)

    tasks = (
        config.OARSI_TASKS.keys()
        if config.multitask_type != "off"
        else ["kl"]
    )

    final_stats = {}

    for task in tasks:
        final_stats[task] = {}
        for m_name in ["acc", "f1", "kappa"]:
            vals = np.array(
                [it[task][m_name] for it in all_iteration_metrics],
                dtype=np.float32
            )

            print(
                f"[Bootstrap] Task={task}, Metric={m_name}, "
                f"Values={vals.tolist()}"
            )

            final_stats[task][m_name] = {
                "values": vals,
                "mean": float(vals.mean()),
                "std": float(vals.std())
            }

    return final_stats

def main(config):
    # ----------------- Setup ----------------- #
    # Copy source files for reproducibility
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

    # Datasets and loaders
    test_ds = KneeMILDataset(config.H5_FILE, test_pids, transform=val_transform)
    test_loader = DataLoader(test_ds, config.BATCH_SIZE, False, collate_fn=mil_collate_fn,
                             num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    # ----------------- Model ----------------- #
    model = get_model(config)
    model_org = get_model_org(config)

    model.load_state_dict(torch.load(
            os.path.join(config.CHECKPOINT_DIR, f"best_model_{config.inference_target}_kappa.pth"), 
            map_location=config.DEVICE
        ))
    if model_org:
        model_org.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH, map_location=config.DEVICE))
        
    
    class_weights_tensor = None
    criterion = get_criterion(config.lossfcn_type, class_weights_tensor, config.OARSI_TASKS)
    optimizer = None

    # ----------------- Training Loop ----------------- #
    test_loss, test_labels, test_preds, test_probs, all_attentions, all_patch_embeddings, all_aggregated_features = run_epoch(
        test_loader, model, model_org, criterion, optimizer, config.DEVICE,
        is_training=False, config=config,
        desc=f"[Testing]"
    )
    print(f"\nTest Loss: {test_loss:.4f}")
    results = [f"Test Loss: {test_loss:.4f}"]
    test_metrics = compute_metrics(config.multitask_type, test_labels, test_preds)

    np.savez(
        os.path.join(config.CHECKPOINT_DIR, "test_pred.npz"),
        id=test_pids,
        prob=test_probs,
        pred=test_preds,
        true_kl=test_labels if config.multitask_type == "off" else test_labels["kl"]
    )
    
    if config.multitask_type == "off":
        task = "kl"
        num_classes = 5
        labels = test_labels
        preds = test_preds
        acc = test_metrics[task]["acc"]
        f1 = test_metrics[task]["f1"]
        kappa = test_metrics[task]["kappa"]
        print(f"[Test] Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
        results.append(f"[Test] Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
        target_names = [f"{task.upper()} {i}" for i in range(num_classes)]

        report = classification_report(labels, preds, target_names=target_names, zero_division=0)
        print(report)
        results.append(f"\n[{task}]\n" + report)
        plt.rcParams.update({
            "font.size": 16,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        })
        disp = ConfusionMatrixDisplay.from_predictions(
            labels, preds, normalize="true", cmap=plt.cm.Greens, values_format='.2f'
        )
        ax = disp.ax_
        for text in ax.texts:
            text.set_fontsize(16)  

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        plt.tight_layout()
        # plt.title(f"{task.upper()} - Normalized Confusion Matrix", fontsize=16)
        plt.savefig(os.path.join(config.CHECKPOINT_DIR, f"cm_{task}_kappa.eps"), format='eps')
        plt.savefig(os.path.join(config.CHECKPOINT_DIR, f"cm_{task}_kappa.png"), format='png')
        plt.close()
        save_path = os.path.join(config.CHECKPOINT_DIR, "inference_result.txt")
        with open(save_path, "w") as f:
            for line in results:
                f.write(line + "\n")
    else:
        
        for task in test_labels.keys():
            labels = test_labels[task]
            preds = test_preds[task]

            acc = test_metrics[task]["acc"]
            f1 = test_metrics[task]["f1"]
            kappa = test_metrics[task]["kappa"]
            print(f"[Test] {task} - Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
            results.append(f"[Test] {task} - Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")

            num_classes = config.OARSI_TASKS[task]
            target_names = [f"{task.upper()} {i}" for i in range(num_classes)]

            report = classification_report(labels, preds, target_names=target_names, zero_division=0)
            print(report)
            results.append(f"\n[{task}]\n" + report)
            
            plt.rcParams.update({
                "font.size": 16,
                "axes.titlesize": 14,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
            })
            disp = ConfusionMatrixDisplay.from_predictions(
                labels, preds, normalize="true", cmap=plt.cm.Greens, values_format='.2f'
            )
            ax = disp.ax_
            for text in ax.texts:
                text.set_fontsize(16)  

            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)

            # Add title
            plt.title(f"{task.upper()} - Normalized Confusion Matrix", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(config.CHECKPOINT_DIR, f"cm_{task}.eps"), format='eps')
            plt.savefig(os.path.join(config.CHECKPOINT_DIR, f"cm_{task}.png"), format='png')
            plt.close()
            save_path = os.path.join(config.CHECKPOINT_DIR, "inference_result.txt")
            with open(save_path, "w") as f:
                for line in results:
                    f.write(line + "\n")

            np.savez(
                os.path.join(config.CHECKPOINT_DIR, f"test_pred_{task}.npz"),
                id=test_pids,
                prob=test_probs[task],
                pred=test_preds[task],
                true=test_labels[task],
                all_attentions=all_attentions[task],
                all_patch_embeddings=all_patch_embeddings[task],
                all_aggregated_features=all_aggregated_features[task],
            )

    print("Inference finished.")

    # ================== Visualization for a single example ==============================
    ######################################################################################
    target_id = "9932578"
    target_side = "R"
    index = np.where(np.array(test_pids)==target_id + "_" + target_side)[0].item()
    target_layer = [model.patch_feature_extractor.conv_block3[0]]
    patches_test, kl_label, id, oarsi_label = test_ds.__getitem__(index)
    
    patch_bag_tensor = torch.stack(patches_test).to(config.DEVICE)  # shape: [41, 1, 16, 16]

    model.eval()
    attention_tool = None
    if config.feedback_type == "off":
        if config.model_type == "MIL_ORG":
            logits, att_scores = model([patch_bag_tensor])
        else:
            logits, att_scores, patch_embeddings, aggregated_features = model([patch_bag_tensor])
    else:
        logits, att_scores, patch_embeddings, aggregated_features = model([patch_bag_tensor], model_org, attention_tool)

    # Argmax across classes
    if config.multitask_type != "off":
        target_classes = logits['kl'].argmax(dim=1)
    else:
        target_classes = logits.argmax(dim=1)
    # print(f"target_classes shape: {target_classes.shape}")      # should be [batch_size]

    # If you want just the first class for score:
    target_class = target_classes[0].item()

    # Use the first logit row (batch item 0) and its predicted class
    if config.multitask_type == "off":
        score = logits[0, target_class]
    else:
        score = logits['kl'][0, target_class]

    model.zero_grad()
    score.backward(retain_graph=True)
    grayscale_cam_dict, PATCH_POINT_INDICES = process_CAM(model, target_layer, target_class, patch_bag_tensor, patches_test, config.CHECKPOINT_DIR)

    ########################################
    data = np.load("./original_data/V00/id_shapes_LR_V00.npz")
    patient_ids = data["id"]
    shapes_L_2d = data["shapes_L"]
    shapes_R_2d = data["shapes_R"]
    kl_grades_L_np = data["KL_L"]
    kl_grades_R_np = data["KL_R"]
    index_test = index
    pid_side = test_pids[index_test]
    pid, side = str.split(pid_side, "_")
    index_all = np.where(np.array(patient_ids) == pid)[0].item()

    visualize = True
    if visualize:
        att_scores = normalize_attention_scores(att_scores.detach().cpu().numpy()) # 41, 1

        visualize_attention_on_img(
            save_path=config.CHECKPOINT_DIR,
            file_path=rf"./original_data/V00/Bilateral_PA_Fixed_Flexion_Knee/{patient_ids[index_all]}.dcm",
            patient_id=patient_ids[index_all],
            index_all=index_all,
            shapes_L_2d=shapes_L_2d,
            shapes_R_2d=shapes_R_2d,
            att_scores=att_scores.squeeze(),  # convert to 1D array
            side=target_side,  # or 'R'
            patchFromPoint=patchFromPoint,
            process_xray=process_xray
        )
        reds_alpha = create_redsalpha()

        visualize_cam_comparisons(
            save_path=config.CHECKPOINT_DIR,
            patient_id=patient_ids[index_all],
            index_all=index_all,
            index_test=index_test,
            side=target_side,  # or "R"
            test_labels=test_labels,
            test_preds=test_preds,
            att_scores=att_scores,
            grayscale_cam_dict=grayscale_cam_dict,
            process_xray_func=process_xray,
            patch_from_point_func=patchFromPoint,
            shapes_L_2d=shapes_L_2d,
            shapes_R_2d=shapes_R_2d,
            file_path_template=f"./original_data/V00/Bilateral_PA_Fixed_Flexion_Knee/{patient_ids[index_all]}.dcm",
            patch_point_indices=PATCH_POINT_INDICES,
            cmap_obj=reds_alpha,
        )

    # ----------------- Bootstrapping ----------------- #
    if config.do_bootstrap:
        bootstrap_stats = bootstrap_evaluation(
            model, model_org, test_pids, config, criterion, n_iterations=5
        )

        print("\n Bootstrapping Statistical Summary (Mean ± SD):")
        bs_results_lines = ["\n--- Bootstrapping Results ---"]
        
        for task, m_dict in bootstrap_stats.items():
            line = f"[{task.upper()}] Acc: {m_dict['acc']['mean']:.4f}±{m_dict['acc']['std']:.4f}, " \
                f"F1: {m_dict['f1']['mean']:.4f}±{m_dict['f1']['std']:.4f}, " \
                f"Kappa: {m_dict['kappa']['mean']:.4f}±{m_dict['kappa']['std']:.4f}"
            print(line)
            bs_results_lines.append(line)

        with open(os.path.join(config.CHECKPOINT_DIR, "inference_result.txt"), "a") as f:
            for line in bs_results_lines:
                f.write(line + "\n")

if __name__ == "__main__":
    from config import build_config
    config_dict = build_config()
    cfg = Config(config_dict)
    cfg.DEBUG_MODE = True
    cfg.WANDB = False
    main(cfg)

    
    