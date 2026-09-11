#!/usr/bin/env python
# coding: utf-8
"""
correlation_medial_lateral_ours_KL_only.py

This is a KL-only adaptation of `correlation_medial_lateral_ours_OARSI_KL.py`.
The pipeline (GradCAM heatmap, attention-weighted medial/lateral intensity,
bonefinder split) is identical; only the model and the prediction step change.

For every patient in `test_pid.json` this script:
    1. Loads their patch bag from the h5 file.
    2. Runs the KL-only model to get a 5-way KL softmax and the MIL
       attention scores.
    3. Computes an *ensemble* GradCAM heatmap (mean of GradCAM, GradCAM++,
       ScoreCAM, AblationCAM, LayerCAM) on the predicted KL class.
    4. Counts how many of the 41 bonefinder patches are to the left vs.
       right of bonefinder point 92, and labels them as lateral/medial
       based on the _L / _R side (see `count_lateral_vs_medial`).
    5. Writes a CSV + JSON results table to `inference/`.
"""

# Imports
# Suppress all tqdm progress bars (used internally by pytorch_grad_cam)
# before any module that imports tqdm gets loaded.
import os
os.environ["TQDM_DISABLE"] = "1"

import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
tqdm.disable = True

from pytorch_grad_cam import (GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, LayerCAM,)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import CompleteMILModel
from dataset import KneeMILDataset, mil_collate_fn


# Constants
H5_FILE         = "./V00_knee_patches_patient_grouped_16_100_all_feature.h5"
MODEL_PATH      = "quantus/models/ours_KL.pth"
MEAN_STD_PATH   = "quantus/utils/mean_std_oai.npy"
TEST_PIDS_JSON  = "test_pid.json" 
SHAPES_NPZ      = "quantus/utils/id_shapeLR_V00.npz"

OUTPUT_DIR      = "inference"
OUTPUT_CSV      = os.path.join(OUTPUT_DIR, "ours_kl_only.csv")
OUTPUT_JSON     = os.path.join(OUTPUT_DIR, "ours_kl_only.json")

KL_NUM_CLASSES = 5

# 41 patch point indices (range1 = medial JS/OS, range2 = lateral JS/OS)
PATCH_POINT_INDICES = np.concatenate([np.arange(9, 27), np.arange(44, 67)])

FEATURE_EXTRACTOR_OUT_DIM = 128
DEFAULT_MAX_PIXEL_VALUE   = 65535.0
BATCH_SIZE                = 1   # we iterate patient-by-patient
DEVICE                    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOT_FRAC = 0.1   # per-patient threshold for "hot" = 10% of max heatmap

# Helpers
def find_last_conv_layer(model: nn.Module) -> str:
    """Return the name of the last nn.Conv2d in the patch feature extractor.

    This replaces the hard-coded `model.patch_feature_extractor.conv_block3[0]`
    used in the original `inference.py`.
    """
    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_name = name
    if last_conv_name is None:
        raise ValueError("No nn.Conv2d layer found in the model.")
    return last_conv_name


def _get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    parts = layer_name.split(".")
    current = model
    for p in parts:
        if p.isdigit():
            current = current[int(p)]
        else:
            current = getattr(current, p)
    return current


def load_mean_std(path: str):
    """Read the training-set mean/std npy; fall back to the hard-coded
    values used in the original `inference.py` if the file is missing
    or malformed.
    """
    fallback = (0.62710685, 0.18273235)
    if not os.path.isfile(path):
        return fallback
    try:
        data = np.load(path, allow_pickle=True)

        if isinstance(data, np.ndarray) and data.dtype == object and data.shape == (2,):
            mean, std = data[0], data[1]
        elif isinstance(data, np.ndarray) and data.shape == (2,) and data.dtype != object:
            mean, std = data[0], data[1]
        else:
            mean, std = data, None

        def _to_float(x):
            if x is None:
                return None
            arr = np.asarray(x)
            if arr.size == 1:
                return float(arr.item())
            return float(arr.flatten()[0])

        m = _to_float(mean)
        s = _to_float(std) if std is not None else None
        if s is None:
            print(f"[correlation_medial_lateral] {path} did not contain a std; using fallback.")
            return fallback
        return m, s
    except Exception as e:
        print(f"[correlation_medial_lateral] Could not load mean/std from {path}: {e}. Using fallback.")
        return fallback


def make_val_transform(mean: float, std: float):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std]),
    ])


def get_ensemble_cam(
    model: nn.Module,
    patch_bag_tensor: torch.Tensor,
    target_class: int,
    target_layer,
) -> np.ndarray:
    """Run the 5 pytorch-grad-cam methods and return their mean.

    Returns:
        ensemble_cam: np.ndarray, shape (N_patches, H_cam, W_cam)
    """
    n_patches = patch_bag_tensor.shape[0]
    targets = [ClassifierOutputTarget(target_class)] * n_patches

    cams = []
    for CamCls in (GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, LayerCAM):
        cam = CamCls(model=model.patch_feature_extractor, target_layers=target_layer)
        grayscale = cam(input_tensor=patch_bag_tensor, targets=targets)
        cams.append(grayscale)

    return np.mean(np.stack(cams, axis=0), axis=0)  # (N_patches, H, W)


def get_split_x(
    shapes_L_at_idx: np.ndarray, shapes_R_at_idx: np.ndarray, side: str) -> float:
    """Compute the vertical-split x-coordinate at the body-centerline landmark.

    The bonefinder point used depends on which knee we are analysing:
        * `_R` knee -> point 92 of the left-knee landmark set
        * `_L` knee -> point 54 of the left-knee landmark set

    Falls back to the right-knee shape if the left-knee shape is too short.
    """
    n_left = len(shapes_L_at_idx)
    point_idx = 92 if side == "R" else 54
    if point_idx < n_left:
        return float(shapes_L_at_idx[point_idx][0])
    return float(shapes_R_at_idx[point_idx - n_left][0])


def count_lateral_vs_medial(pid_side: str, shapes_L_at_idx: np.ndarray, shapes_R_at_idx: np.ndarray, patch_point_indices: np.ndarray, att_scores=None, ensemble_cam=None):
    """Sum the attention-weighted ensemble-CAM intensity AND the raw number
    of hot pixels on each side of the bonefinder split, and report which
    side is dominant.

    For a *right* knee (pid_side ends with `_R`):
        center_x < split_x -> LATERAL  (outer edge, away from body center)
        center_x > split_x -> MEDIAL   (inner edge, toward body center)

    For a *left* knee (pid_side ends with `_L`):
        center_x < split_x -> MEDIAL
        center_x > split_x -> LATERAL

    Per-patch intensity (lateral_intensity / medial_intensity):
        MIL attention score * mean(ensemble_cam[i]).
    Hot-pixel count (lateral_hot_count / medial_hot_count):
        per-pixel count of `ensemble_cam[i] >= 0.1 * max(ensemble_cam)`
        summed across patches in that side (so a 16x16 patch contributes
        0..256 hot pixels to its side).

    Returns:
        side (str),
        lateral_intensity, medial_intensity (float),
        lateral_hot_count, medial_hot_count (int),
        dominant (str), split_x (float)
    """

    side = pid_side[-1]                                # 'L' or 'R'
    pts  = shapes_R_at_idx if side == "R" else shapes_L_at_idx
    split_x = get_split_x(shapes_L_at_idx, shapes_R_at_idx, side)

    # Normalize att_scores to a 1-D tensor of length N_patches.
    if att_scores is not None:
        att_scores = att_scores.squeeze()
        if att_scores.ndim == 2:
            att_scores = att_scores[0]
        att_scores_list = [float(att_scores[i].item()) for i in range(att_scores.shape[0])]
    else:
        att_scores_list = None

    # Pre-compute the hot-pixel threshold (per patient).
    if ensemble_cam is not None:
        max_cam = float(ensemble_cam.max()) + 1e-12
        hot_threshold = HOT_FRAC * max_cam
    else:
        hot_threshold = None

    lateral_intensity     = 0.0
    medial_intensity      = 0.0
    lateral_hot_count     = 0
    medial_hot_count      = 0

    for i, point_idx in enumerate(patch_point_indices):
        center_x = float(pts[point_idx][0])
        if side == "R":
            is_lateral = center_x < split_x
        else:  # 'L'
            is_lateral = center_x > split_x

        if att_scores_list is not None and ensemble_cam is not None:
            w = att_scores_list[i] * float(ensemble_cam[i].mean())
        else:
            w = 1.0   # plain-count fallback

        if is_lateral:
            lateral_intensity += w
        else:
            medial_intensity  += w

        if hot_threshold is not None:
            n_hot_in_patch = int((ensemble_cam[i] >= hot_threshold).sum())
            if is_lateral:
                lateral_hot_count += n_hot_in_patch
            else:
                medial_hot_count  += n_hot_in_patch

    if lateral_intensity > medial_intensity:
        dominant = "lateral"
    elif medial_intensity > lateral_intensity:
        dominant = "medial"
    else:
        dominant = "tie"

    return (side, lateral_intensity, medial_intensity,
            lateral_hot_count, medial_hot_count,
            dominant, split_x)


# Main
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load test PIDs 
    with open(TEST_PIDS_JSON, "r") as f:
        test_pids = json.load(f)
    print(f"[correlation_medial_lateral] {len(test_pids)} test patients from {TEST_PIDS_JSON}")

    # Load bonefinder shapes 
    shapes_data   = np.load(SHAPES_NPZ, allow_pickle=True)
    patient_ids   = list(shapes_data["id"])
    pid_to_idx    = {pid: i for i, pid in enumerate(patient_ids)}
    shapes_L      = shapes_data["shapes_L"]   # (N, n_left, 2)
    shapes_R      = shapes_data["shapes_R"]   # (N, n_right, 2)
    print(f"[correlation_medial_lateral] {len(patient_ids)} patients in {SHAPES_NPZ}")

    # Load mean / std 
    mean, std = load_mean_std(MEAN_STD_PATH)
    print(f"[correlation_medial_lateral] Using mean={mean:.6f}, std={std:.6f}")

    val_transform = make_val_transform(mean, std)

    # Build dataset + model 
    test_dataset = KneeMILDataset(
        H5_FILE,
        test_pids,
        transform=val_transform,
        max_pixel_value=DEFAULT_MAX_PIXEL_VALUE,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=mil_collate_fn,
        num_workers=0,
    )

    # KL-only model: 5-way classifier head (KL grades 0-4).
    model = CompleteMILModel(
        feature_extractor_out_dim=FEATURE_EXTRACTOR_OUT_DIM,
        num_classes=KL_NUM_CLASSES,
        aggregation_type="attention",
    ).to(DEVICE)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"[correlation_medial_lateral] Loaded KL-only model weights from {MODEL_PATH}")

    target_layer_name = find_last_conv_layer(model.patch_feature_extractor)
    target_layer = [_get_layer_by_name(model.patch_feature_extractor, target_layer_name)]
    print(f"[correlation_medial_lateral] GradCAM target layer: patch_feature_extractor.{target_layer_name}")

    # Inference loop
    rows = []
    summary = {"medial": 0, "lateral": 0, "tie": 0, "skipped": 0}

    # NOTE: autograd must stay ON here -- pytorch_grad_cam calls
    # `loss.backward()` internally. We do NOT wrap the loop in
    # `torch.no_grad()`. `model.eval()` only affects dropout / BN.


    for batch in test_loader:
        if batch is None or batch[0] is None:
            summary["skipped"] += 1
            continue

        patch_bags, _kl_labels, ids_batch, _features = batch
        pid_side = ids_batch[0]

        # Some samples may be filtered out by KneeMILDataset; if the
        # requested pid_side is not in the dataset we just skip it.
        if pid_side not in test_pids:
            summary["skipped"] += 1
            continue

        # Reset gradients before each forward
        model.zero_grad()
        patch_bag_tensor = patch_bags[0].to(DEVICE)              # (N, 1, 16, 16)

        # 1) Forward pass -> KL logits (1, 5) + attention scores.
        #    The KL-only CompleteMILModel returns a 4-tuple:
        #    (final_batch_logits, att_scores, patch_embeds, agg_feats)
        #    Unlike the multitask variant there is no per-task dict.
        outputs = model([patch_bag_tensor])
        if outputs is None:
            summary["skipped"] += 1
            continue
        kl_logits, att_scores, _patch_feats, _agg_feats = outputs

        # 2) Predicted KL class (used as the GradCAM target)
        target_class = int(torch.argmax(kl_logits[0]).item())

        # 3) Ensemble CAM (5 methods, mean) -- used below to weight
        #    the per-patch intensity on each side of the bonefinder split.
        ensemble_cam = get_ensemble_cam(
            model, patch_bag_tensor, target_class, target_layer
        )

        # 4) KL prediction
        pred_kl = int(torch.argmax(kl_logits[0]).item())

        def _to_gt(v):
            v = int(v)
            return v if v != -999 else None

        gt_kl   = int(_kl_labels[0].item())
        gt_jsnm = _to_gt(_features[0, 0, 0].item())
        gt_jsnl = _to_gt(_features[0, 0, 1].item())
        gt_osfm = _to_gt(_features[0, 0, 2].item())
        gt_ostm = _to_gt(_features[0, 0, 3].item())
        gt_osfl = _to_gt(_features[0, 0, 4].item())
        gt_ostl = _to_gt(_features[0, 0, 5].item())

        # Derived GT signed differences (missing sub-scores treated as 0)
        gt_jsn_ml = (gt_jsnm or 0) - (gt_jsnl or 0)
        gt_osn_ml = ((gt_osfm or 0) + (gt_ostm or 0)) \
                    - ((gt_osfl or 0) + (gt_ostl or 0))


        # 5) Locate patient in bonefinder shapes and count
        bare_pid = pid_side[:-2]   # strip "_L" or "_R"
        if bare_pid not in pid_to_idx:
            print(f"  [warn] {pid_side}: bare pid {bare_pid} not in shapes npz. Skipping.")
            summary["skipped"] += 1
            continue
        idx_in_shapes = pid_to_idx[bare_pid]

        (side, lateral_intensity, medial_intensity,
         lateral_hot_count, medial_hot_count,
         dominant, split_x) = count_lateral_vs_medial(
            pid_side,
            shapes_L[idx_in_shapes],
            shapes_R[idx_in_shapes],
            PATCH_POINT_INDICES,
            att_scores=att_scores,
            ensemble_cam=ensemble_cam,
        )
        summary[dominant] = summary.get(dominant, 0) + 1

        # Signed M - L difference of the per-patch heatmap intensity
        # (attention * mean(ensemble_cam)).
        heatmap_ml = medial_intensity - lateral_intensity

        # Raw pixel-level hot counts: each patch is 16x16=256 pixels, so the
        # max possible hot pixels per side is roughly 256 * n_patches_on_side.
        hot_count_ml = medial_hot_count - lateral_hot_count

        # NOTE: The KL-only model cannot predict OARSI sub-grades.
        # We record `None` for those columns (and for the derived
        # `jsn_ml` / `osn_ml`) to make it explicit in the CSV.
        row = {
            "pid_side":          pid_side,
            "side":              side,
            "n_patches":         int(len(PATCH_POINT_INDICES)),
            "split_x":           round(split_x, 2),
            "lateral_intensity": round(lateral_intensity, 6),
            "medial_intensity":  round(medial_intensity,  6),
            "lateral_hot_count": lateral_hot_count,
            "medial_hot_count":  medial_hot_count,
            "hot_count_ml":      hot_count_ml,
            "dominant_side":     dominant,
            "heatmap_ml":        round(heatmap_ml, 6),
            "jsn_ml":            None,            # KL-only model -> unavailable
            "osn_ml":            None,            # KL-only model -> unavailable
            "pred_kl":           pred_kl,
            "pred_jsnm":         None,            # KL-only model -> unavailable
            "pred_jsnl":         None,            # KL-only model -> unavailable
            "pred_osfm":         None,            # KL-only model -> unavailable
            "pred_ostm":         None,            # KL-only model -> unavailable
            "pred_ostl":         None,            # KL-only model -> unavailable
            "pred_osfl":         None,            # KL-only model -> unavailable
            "gt_kl":             gt_kl,
            "gt_jsnm":           gt_jsnm,
            "gt_jsnl":           gt_jsnl,
            "gt_jsn_ml":         gt_jsn_ml,
            "gt_osfm":           gt_osfm,
            "gt_ostm":           gt_ostm,
            "gt_osfl":           gt_osfl,
            "gt_ostl":           gt_ostl,
            "gt_osn_ml":         gt_osn_ml,
        }
        rows.append(row)
        print(
            f"  {pid_side:>12s}  split_x={split_x:8.1f}  "
            f"L={lateral_intensity:7.4f}  M={medial_intensity:7.4f}  "
            f"ML={heatmap_ml:+7.4f}  "
            f"hot_L={lateral_hot_count:4d}  hot_M={medial_hot_count:4d}  "
            f"hot_ML={hot_count_ml:+4d}  -> {dominant:7s}  "
            f"pred_kl={pred_kl}"
        )


    # Write outputs 
    fieldnames = list(rows[0].keys()) if rows else [
        "pid_side", "side", "n_patches", "split_x",
        "lateral_intensity", "medial_intensity",
        "lateral_hot_count", "medial_hot_count", "hot_count_ml",
        "dominant_side",
        "heatmap_ml", "jsn_ml", "osn_ml",
        "pred_kl", "pred_jsnm", "pred_jsnl",
        "pred_osfm", "pred_ostm", "pred_ostl", "pred_osfl",
        "gt_kl", "gt_jsnm", "gt_jsnl", "gt_jsn_ml",
        "gt_osfm", "gt_ostm", "gt_osfl", "gt_ostl", "gt_osn_ml",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[correlation_medial_lateral] Wrote {len(rows)} rows -> {OUTPUT_CSV}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({r["pid_side"]: r for r in rows}, f, indent=2)
    print(f"[correlation_medial_lateral] Wrote JSON           -> {OUTPUT_JSON}")

    # Summary 
    print("\n[correlation_medial_lateral] Summary across test set:")
    for k in ("medial", "lateral", "tie", "skipped"):
        print(f"  {k:8s}: {summary.get(k, 0)}")


if __name__ == "__main__":
    main()

