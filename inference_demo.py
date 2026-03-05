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

    train = []
    val = []
    test = np.array(groups)

    print(f"Total inference samples: {len(test)}")


    train_pids, val_pids, test_pids = [], [], test.tolist()
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
        
    # ----------------- Loss & Optimizer ----------------- #   
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
    
    if config.multitask_type == "off":
        print(f"Test Labels: {[x.item() for x in test_labels]}")
        print(f"Test Preds: {[x.item() for x in test_preds]}")
    else:
        test_labels = {
            k: [x.item() if hasattr(x, "item") else x for x in v]
            for k, v in test_labels.items()
        }

        test_preds = {
            k: [x.item() if hasattr(x, "item") else x for x in v]
            for k, v in test_preds.items()
        }
        print("test labels:", test_labels)
        print("test preds:", test_preds)


    # # ================== Visualization for a single example ==============================
    # ######################################################################################
    target_id = "9008884"
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

    if config.multitask_type != "off":
        target_classes = logits['kl'].argmax(dim=1)
    else:
        target_classes = logits.argmax(dim=1)

    target_class = target_classes[0].item()

    if config.multitask_type == "off":
        score = logits[0, target_class]
    else:
        score = logits['kl'][0, target_class]

    model.zero_grad()
    score.backward(retain_graph=True)
    grayscale_cam_dict, PATCH_POINT_INDICES = process_CAM(model, target_layer, target_class, patch_bag_tensor, patches_test, config.CHECKPOINT_DIR)

    data = np.load("./original_data/V00/id_shapes_LR_V00.npz")
    patient_ids = data["id"]
    shapes_L_2d = data["shapes_L"]
    shapes_R_2d = data["shapes_R"]
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

if __name__ == "__main__":
    from config import build_config
    config_dict = build_config()
    cfg = Config(config_dict)
    cfg.DEBUG_MODE = True
    cfg.WANDB = False
    main(cfg)

    
    