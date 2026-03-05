from html import parser
import os
import torch
import argparse
from datetime import datetime


# ---------------- Default Configuration ---------------- #
NOW = datetime.now().strftime('%Y%m%d_%H%M%S')

DATA_HALF = False

# Dataset / Checkpoints
DEFAULT_H5_FILE = "knee_patches_patient_grouped_16_100_all_feature.h5"
# DEFAULT_H5_FILE = "./original_data/V00/V00_knee_patches_patient_grouped_16_128_all_feature_roundjsn.h5"
DEFAULT_PRE_CKPT_DIR = "model_checkpoints_tnc_final"
DEFAULT_PRETRAINED_MODEL = os.path.join(DEFAULT_PRE_CKPT_DIR, "best_model_val_kappa.pth")

# Training hyperparameters
KL_NUM_CLASSES = 5
OARSI_TASKS = {
            "jsnm": 4,  # 0–3 ordinal
            "jsnl": 4,  # 0–3 ordinal
            "osfm": 4,  # 0–3 ordinal
            "ostm": 4,  # 0–3 ordinal
            "ostl": 4,  # 0–3 ordinal
            "osfl": 4,  # 0–3 ordinal
        }
NUM_FEATURES = len(OARSI_TASKS)

FEATURE_EXTRACTOR_OUT_DIM = 128
AGGREGATION_TYPE = "attention"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 200
# SEED = 42
DEFAULT_MAX_PIXEL_VALUE = 65535.0

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda" and NUM_WORKERS > 0


# ---------------- Argument Parser ---------------- #
def get_args():
    parser = argparse.ArgumentParser()
    # Experiment setup
    parser.add_argument("--current_ckpt", type=str, default=None)
    parser.add_argument("--use_baseline", action="store_true", help="Use baseline MIL model")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
            "--model_type",
            type=str,
            choices=["MIL", "MIL_MultiTask", "MIL_MultiTask_SharedHead", "MIL_wGP_MultiTask", "MILOrdinal",
                    "MILOrdinal_MultiTask", "MILCoral_MultiTask", "MIL_MultiTask_imedslab", "MILOrdinal_MultiTask_imedslab"],
            default="MIL",
            help= "Choose the MIL model type"
        )

    # Loss function type
    parser.add_argument(
        "--lossfcn_type",
        type=str,
        choices=["CrossEntropy","CrossEntropy_MultiTask", "CoralLoss_MultiTask", "CoralLossWeighted", "CoralFocalLoss_MultiTask", 
                 "CoralLossEffective", "CoralFocalLoss_MultiTask_MetricsBalanced", "BCEWithLogitsLoss_MultiTask"],
        default="CrossEntropy",
        help="Choose the loss function"
    )

    parser.add_argument(
        "--predict_criteria",
        type=str, 
        choices=["Max", "Max_Multitask", "Coral_Multitask", "Coral", "ordinal"],
        default="Max"
    )

    # Multitask type
    parser.add_argument(
        "--multitask_type",
        type=str,
        choices=["off", "kl_jsn", "all"],
        default="off",
        help="Choose the tpye of multitask"
    )

    parser.add_argument(
        "--classweight_type",
        type=str,
        choices = ["inv", "effective", "balanace_sampling", "all_metrics_inv"], # effective has positive and negative
        default="inv"

    )
    
    # feedback type
    parser.add_argument(
        "--feedback_type",
        type=str,
        choices=["off", "on"],
        default="off",
        help="whether use feedback"
    )

    # CAM type
    parser.add_argument(
        "--feedback_cam",
        type=str,
        choices=["off", "GradCAM", "GradCAMPlusPlus", "ScoreCAM", "AblationCAM", "LayerCAM"],
        default="off",
        help= "Choose the CAM method, select 'off' if not using CAM"
    )

    # pretrained checkpoint path
    parser.add_argument(
        "--pre_ckpt",
        type=str,
        default=None,
        help="Path to pretrained checkpoint dir (required if feedback_type is 'on')"
    )
    parser.add_argument(
        "--inference_target",
        type=str,
        default="kl"
    )
    parser.add_argument(
        "--balance_sampling",
        action="store_true", 
        help="Use balance sampling"
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Additional note for experiment naming"
    )

    parser.add_argument(
        "--patient_id",
        type=str,
        default="",
        help="Patient ID for inference"
    )

    args = parser.parse_args()

    if args.feedback_type == "on" and not args.pre_ckpt:
        parser.error("feedback_type='on' requires --pre_ckpt to be specified.")
    
    
    return args


# ---------------- Build Final Config ---------------- #
def build_config():
    args = get_args()

    # number of features (you can adjust this logic if dynamic)
    num_features = 6

    # dataset info
    data_part = "halfdata" if DATA_HALF else "wholedata"

    # timestamp + run_name

    loss_map = {"CrossEntropy": "CE", "CrossEntropy_MultiTask": "CEoM", "CoralLoss_MultiTask": "CLoM", "CoralLossWeighted": "CLoW", 
                "CoralFocalLoss_MultiTask": "CFLoM", "CoralLossEffective": "CLoE", "CoralFocalLoss_MultiTask_MetricsBalanced": "CFLMB",
                "OrdinalMSE": "MSE", "BCEWithLogitsLoss_MultiTask": "BCEWL"}
    mtask_map = {"off": "0", "kl_jsn": "KJ", "all": "A"}
    cam_map = {"off": "0", "GradCAM": "GC", "GradCAMPlusPlus": "GPP", 
            "ScoreCAM": "SC", "AblationCAM": "AC", "LayerCAM": "LC"}

    run_name = (
        f"{NOW}_{args.model_type}"
        f"_L{loss_map[args.lossfcn_type]}"
        f"_M{mtask_map[args.multitask_type]}"
        f"_C{cam_map[args.feedback_cam]}"
        f"_F{args.feedback_type[0]}"   # o / n
        f"_lr{LEARNING_RATE:.0e}_b{BATCH_SIZE}"
        f"_{args.note}"
    )

    # checkpoint dir
    if args.current_ckpt:
        checkpoint_dir = os.path.join(
            "original_data",
            "V00",
            args.current_ckpt
        )

    else:
        checkpoint_dir = os.path.join(
            "original_data",
            "V00",
            f"model_checkpoints_{NOW}_epoch{NUM_EPOCHS}_{args.model_type}_L{loss_map[args.lossfcn_type]}_M{mtask_map[args.multitask_type]}_C{cam_map[args.feedback_cam]}_F{args.feedback_type[0]}_lr{LEARNING_RATE:.0e}_b{BATCH_SIZE}"
        )

    if args.multitask_type == "all":
        OARSI_TASKS  = {
            "kl": 5, "jsnm": 4, "jsnl": 4, "osfm": 4, "ostm": 4, "ostl": 4, "osfl": 4
        }
    elif args.multitask_type == "kl_jsn":
        OARSI_TASKS = {
            "kl": 5, "jsnm": 4, "jsnl": 4
        }
    else:
        OARSI_TASKS = {"kl": 5}

    config = {
        # experiment setup
        "DEBUG_MODE": args.debug,
        "WANDB": not args.debug,
        "DATA_HALF": DATA_HALF,

        # dataset paths
        "H5_FILE": DEFAULT_H5_FILE,
        "PRE_CHECKPOINT_DIR": args.pre_ckpt,
        "CHECKPOINT_DIR": checkpoint_dir,
        "MEAN_STD_FILE_PATH": os.path.join(checkpoint_dir, "mean_std_train_patches.npy"),
        "PRETRAINED_MODEL_PATH": DEFAULT_PRETRAINED_MODEL,

        # hyperparameters
        "FEATURE_EXTRACTOR_OUT_DIM": FEATURE_EXTRACTOR_OUT_DIM,
        "AGGREGATION_TYPE": AGGREGATION_TYPE,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "SEED": args.seed,
        "DEFAULT_MAX_PIXEL_VALUE": DEFAULT_MAX_PIXEL_VALUE,
        "NUM_FEATURES": num_features,
        "DATA_PART": data_part,

        # tasks
        "KL_NUM_CLASSES": KL_NUM_CLASSES,
        "OARSI_TASKS": OARSI_TASKS,

        # Ablation study
        "model_type": args.model_type, # "MIL", "MILOrdinal", "MILOrdinal_MultiTask"
        "lossfcn_type": args.lossfcn_type, # "CoralLossWeighted", "OrdinalAndFocal", "OrdinalMSE", 
        "multitask_type": args.multitask_type, # "off", "kl", "kl_jsn", "all"
        "feedback_type": args.feedback_type, # "off", "on"
        "feedback_cam": args.feedback_cam, # "GradCAM", "GradCAMPlusPlus", "ScoreCAM", "AblationCAM", "LayerCAM"
        "classweight_type": args.classweight_type, 
        "predict_criteria": args.predict_criteria,
        "inference_target": args.inference_target,
        "balance_sampling": args.balance_sampling,

        # device
        "DEVICE": DEVICE,
        "NUM_WORKERS": NUM_WORKERS,
        "PIN_MEMORY": PIN_MEMORY,
        
        # info
        "NOW": NOW,
        "run_name": run_name,
        "note": args.note,
        "patient_id": args.patient_id
    }

    # make dirs if needed
    os.makedirs(config["CHECKPOINT_DIR"], exist_ok=True)

    return config
