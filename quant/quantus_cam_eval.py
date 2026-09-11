#!/usr/bin/env python
# coding: utf-8

"""
Quantus CAM Evaluation - Ours (ensemble CAM) vs Tiulpin 2018 (GradCAM on Branch)

Single radar chart comparing two explanation approaches across 4 metrics:
Complexity (Sparseness), Faithfulness (FaithfulnessCorrelation),
Robustness (AvgSensitivity), Randomisation (RandomLogit).
"""
import os
os.environ["TQDM_DISABLE"] = "1"
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import quantus
import random

# SSIM patch
import skimage.metrics
# Pearson correlation patch (NaN-safe, handles constant pred_deltas from white-flood perturbation)
import scipy.stats
from pytorch_grad_cam import ( GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, LayerCAM,)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
sns.set_theme()

from quant.quant_utils import (KneeNet, TiulpinWrapper, MILWrapper, build_our_model, load_our_sample, load_tiulpin_batch, find_common_samples, TIULPIN_DATA_ROOT, TIULPIN_MODEL_PATH, OURS_H5_FILE, OURS_MODEL_PATH)

# Multi-task (KL + OARSI) variant of our MIL model. Uses the same
# `MIL_MultiTask_imedslab` architecture as the single-task `build_our_model`
# (head_0 = KL 5-class), with additional OARSI heads (head_1..head_6).
OURS_OARSI_KL_MODEL_PATH = "quantus/models/ours_OARSI_KL.pth"

# Constants
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NR_CHANNELS = 1
NUM_CLASSES = 5
DEFAULT_MAX_PIXEL_VALUE = 65535.0
MAX_SAMPLES = 100  # limit to first N common samples to keep runtime manageable

# Ours image size (patches are 16x16)
OURS_IMG_SIZE = 16
# Tiulpin image size (patches are 128x128)
TIULPIN_IMG_SIZE = 128


# Monkey-patches (same fixes as quantus_old.py)
import functools
from quantus.metrics.faithfulness.faithfulness_correlation import FaithfulnessCorrelation
from quantus.helpers import warn as _fc_warn


def _class_pred_scalar(predict_fn, x_input, y):
    preds = np.asarray(predict_fn(x_input))
    if preds.ndim == 0:
        return float(preds)
    while preds.ndim > 1 and preds.shape[0] == 1:
        preds = preds[0]
    if preds.ndim == 1:
        return float(preds[y])
    raise ValueError(f"Unexpected prediction shape: {preds.shape}")


DEBUG_INSTANCE = {"count": 0}

def _patched_evaluate_instance(self, model, x, y, a, **kwargs):
    a = a.flatten()
    x_input = model.shape_input(x, x.shape, channel_first=True)
    y_pred = _class_pred_scalar(model.predict, x_input, y)

    perturb_baseline = getattr(self, "perturb_baseline", "black")
    pred_deltas, att_sums = [], []

    for _ in range(self.nr_runs):
        a_ix = np.random.choice(a.shape[0], self.subset_size, replace=False)
        pf = self.perturb_func

        if isinstance(pf, functools.partial):
            bound = dict(pf.keywords or {})
            bound["perturb_baseline"] = perturb_baseline
            x_perturbed = pf(arr=x, indices=a_ix,
                             indexed_axes=self.a_axes, **bound)
        else:
            x_perturbed = pf(arr=x, indices=a_ix,
                             indexed_axes=self.a_axes,
                             perturb_baseline=perturb_baseline)
            
        _fc_warn.warn_perturbation_caused_no_change(x=x, x_perturbed=x_perturbed)
        x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
        y_pred_perturb = _class_pred_scalar(model.predict, x_input, y)
        pred_deltas.append(float(y_pred - y_pred_perturb))
        att_sums.append(np.sum(a[a_ix]))

    # DEBUG
    if DEBUG_INSTANCE["count"] < 5:
        print(f"\n[instance {DEBUG_INSTANCE['count']}] y={y} y_pred={y_pred:.4f}")
        print(f"  a stats: min={a.min():.4f} max={a.max():.4f} std={a.std():.4f}")
        print(f"  pred_deltas: {pred_deltas}")
        print(f"  att_sums:    {att_sums}")
        DEBUG_INSTANCE["count"] += 1
    # END DEBUG

    return self.similarity_func(a=att_sums, b=pred_deltas)


FaithfulnessCorrelation.evaluate_instance = _patched_evaluate_instance

def _patched_ssim(a, b, **kwargs):
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    combined = np.concatenate([a_arr.ravel(), b_arr.ravel()]) if a_arr.size and b_arr.size else np.array([0.0])
    data_range = float(combined.max() - combined.min())
    if data_range == 0.0:
        data_range = 1.0
    user_range = kwargs.pop("data_range", None)
    if user_range is not None:
        data_range = user_range
    win_size = kwargs.pop("win_size", None)
    return skimage.metrics.structural_similarity(
        im1=a_arr, im2=b_arr, win_size=win_size, data_range=data_range, **kwargs
    )

quantus.similarity_func.ssim = _patched_ssim
try:
    quantus.functions.similarity_func.ssim = _patched_ssim
except AttributeError:
    pass

def _safe_pearson(a, b, **kwargs):
    """NaN-safe Pearson correlation: returns 0.0 if either input is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(scipy.stats.pearsonr(a, b)[0])


# Old API (just in case anything still uses it)
quantus.similarity_func.correlation_pearson = _safe_pearson

# New API — this is the one that actually gets called by FaithfulnessCorrelation
try:
    quantus.functions.similarity_func.pearson_correlation = _safe_pearson
except AttributeError:
    pass

# Safe normalise helper
def safe_normalise_by_negative(a, normalise_axes=None):
    a = np.asarray(a, dtype=np.float64)
    a_max = a.max()
    a_min = a.min()
    out = np.zeros_like(a)
    if a_max > 0:
        pos_mask = a > 0
        out[pos_mask] = a[pos_mask] / a_max
    if a_min < 0:
        neg_mask = a < 0
        out[neg_mask] = a[neg_mask] / abs(a_min)
    return out

# EXPLAIN FUNCTIONS FOR QUANTUS

# Ours: ensemble CAM via process_CAM logic 

def _find_last_conv_layer(model):
    """Return the last Conv2d layer name in a sequential/recursive model."""
    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_name = name
    if last_conv_name is None:
        raise ValueError("No Conv2d layer found in model.")
    return last_conv_name


def _get_layer_by_name(model, layer_name):
    """Retrieve a sub-module by dotted name (handles integer indices for Sequential)."""
    parts = layer_name.split(".")
    current = model
    for p in parts:
        if p.isdigit():
            current = current[int(p)]
        else:
            current = getattr(current, p)
    return current


def ours_cam_explainer(model, inputs, targets, **kwargs) -> np.ndarray:
    """
    Quantus explain_func for our MIL model using ensemble CAM
    (average of GradCAM, GradCAM++, ScoreCAM, AblationCAM, LayerCAM).

    Returns explanation as np.ndarray of shape (B, H, W).
    """

    gc.collect()
    torch.cuda.empty_cache()

    device_obj = kwargs.get("device", device)
    model = model.to(device_obj)
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32, device=device_obj)
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets, dtype=torch.long, device=device_obj)

    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(1)

    internal = getattr(model, "mil_model", model)
    feature_extractor = internal.patch_feature_extractor
    target_layer_name = _find_last_conv_layer(feature_extractor)

    if hasattr(model, "mil_model"):
        full_target_path = f"mil_model.patch_feature_extractor.{target_layer_name}"
    else:
        full_target_path = f"patch_feature_extractor.{target_layer_name}"

    target_layer = [_get_layer_by_name(model, full_target_path)]
    cam_targets = [ClassifierOutputTarget(int(t.item())) for t in targets]

    cam_classes = [GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, LayerCAM]
    cam_maps = []

    for cam_cls in cam_classes:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            cam = cam_cls(model=model, target_layers=target_layer)
            # ScoreCAM / AblationCAM don't need gradients but pytorch_grad_cam
            # handles that internally — just call the same way.
            result = cam(input_tensor=inputs, targets=cam_targets).astype(np.float64)
            cam_maps.append(result)
            del cam
        except Exception as e:
            print(f"  [!] {cam_cls.__name__} failed, skipping: {e}")
        gc.collect()
        torch.cuda.empty_cache()

    if not cam_maps:
        raise RuntimeError("All CAM methods failed for ours_cam_explainer.")

    # Average across whichever CAM methods succeeded
    ensemble_result = np.mean(np.stack(cam_maps, axis=0), axis=0)

    gc.collect()
    torch.cuda.empty_cache()
    return ensemble_result

# Tiulpin 2018: GradCAM on the Branch net
def tiulpin_gradcam_explainer(model, inputs, targets, **kwargs) -> np.ndarray:
    """
    Quantus explain_func for Tiulpin KneeNet using pytorch_grad_cam.GradCAM
    on the last Conv layer of the shared Branch sub-network.

    Returns explanation as np.ndarray of shape (B, H, W).
    """

    gc.collect()
    torch.cuda.empty_cache()

    device_obj = kwargs.get("device", device)
    model = model.to(device_obj)
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32, device=device_obj)
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets, dtype=torch.long, device=device_obj)

    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(1)

    # Find target layer within kneenet.branch, but pass the full model to
    # GradCAM so backprop through the final FC layer produces valid gradients.
    kneenet = getattr(model, "kneenet", model)
    target_layer_name = _find_last_conv_layer(kneenet.branch)

    if hasattr(model, "kneenet"):
        full_target_path = f"kneenet.branch.{target_layer_name}"
    else:
        full_target_path = f"branch.{target_layer_name}"

    target_layer = [_get_layer_by_name(model, full_target_path)]

    cam = GradCAM(model=model, target_layers=target_layer)
    cam_targets = [ClassifierOutputTarget(int(t.item())) for t in targets]

    result = cam(input_tensor=inputs, targets=cam_targets).astype(np.float64)

    gc.collect()
    torch.cuda.empty_cache()
    return result


# RADAR CHART 

def radar_factory(num_vars, frame="circle"):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, angles=None):
            self.set_thetagrids(angles=np.degrees(theta), labels=labels)

        def _gen_axes_patch(self):
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            if frame == "polygon":
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                spine = Spine(axes=self, spine_type="circle",
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5)
                                    + self.transAxes)
                return {"polar": spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

# MAIN
def _build_metrics(img_size=None, perturb_fraction=0.10):
    """Create fresh Quantus metric instances (re-created per sample for safety).
    
    Parameters
-
    img_size : int or None
        Spatial dimension of the square input images (e.g. 16 for Ours, 128 for Tiulpin).
        If None, the returned dict is only suitable for key iteration, not evaluation.
    perturb_fraction : float
        Fraction of pixels to perturb for FaithfulnessCorrelation.
    """
    if img_size is not None:
        subset_size = max(2, int(perturb_fraction * img_size * img_size))
    else:
        subset_size = 1  # placeholder, never actually used for evaluation
    
    return {
        "Complexity": quantus.Sparseness(
            abs=True,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Faithfulness": quantus.FaithfulnessCorrelation(
            nr_runs=10,
            subset_size=subset_size,
            perturb_baseline="white",
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            # perturb_func=quantus.perturb_func.gaussian_noise,
            perturb_func_kwargs={"perturb_std": 0.1},  
            similarity_func=quantus.similarity_func.correlation_pearson,
            abs=False,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Robustness": quantus.AvgSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            perturb_func=quantus.perturb_func.uniform_noise,
            similarity_func=quantus.similarity_func.difference,
            abs=False,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Randomisation": quantus.RandomLogit(
            num_classes=NUM_CLASSES,
            similarity_func=quantus.similarity_func.ssim,
            abs=True,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
        ),
    }


def main():
    samples = find_common_samples(OURS_H5_FILE, TIULPIN_DATA_ROOT)[:MAX_SAMPLES]
    if not samples:
        print("ERROR: No common samples found between HDF5 and Tiulpin datasets.")
        return None

    print(f"\nEvaluating on {len(samples)} common sample(s).")

    # 1. Load models ONCE

    # Ours model (single-task KL)
    print("\n[Ours] Building model …")
    mil_model = build_our_model(OURS_MODEL_PATH, device)
    ours_model = MILWrapper(mil_model, num_classes=NUM_CLASSES).to(device)
    ours_model.eval()

    # Ours model (multi-task KL+OARSI)
    print("\n[Ours (KL+OARSI)] Building multi-task model …")
    oarsi_kl_mil_model = build_our_model(OURS_OARSI_KL_MODEL_PATH, device)
    oarsi_kl_model = MILWrapper(oarsi_kl_mil_model, num_classes=NUM_CLASSES).to(device)
    oarsi_kl_model.eval()

    # Tiulpin model
    print("\n[Tiulpin] Building model …")
    kneenet = KneeNet(bw=32, drop=0.2).to(device)
    if os.path.isfile(TIULPIN_MODEL_PATH):
        kneenet.load_state_dict(torch.load(TIULPIN_MODEL_PATH, map_location=device))
        print(f"  Loaded weights from {TIULPIN_MODEL_PATH}")
    else:
        print(f"  WARNING: model file not found at '{TIULPIN_MODEL_PATH}'.")
    tiulpin_model = TiulpinWrapper(kneenet).to(device)
    tiulpin_model.eval()

    # 2. Evaluate each sample and accumulate per-sample scores
    # Per-method, per-metric lists to store one scalar per sample
    method_names = [
        "Ours (CAM, KL)",
        "Ours (CAM, KL+OARSI)",
        "Tiulpin 2018 (GradCAM)",
    ]
    accum = {m: {metric: [] for metric in _build_metrics()} for m in method_names}

    for s_idx, (sample_id, sample_side, sample_key) in enumerate(samples):
        print(f"\n{'=' * 60}")
        print(f"Sample {s_idx + 1}/{len(samples)}: {sample_key}")
        print(f"{'=' * 60}")

        # Load "Ours" patches
        print("  [Ours] Loading patches …")
        patches_np, kl_true = load_our_sample(OURS_H5_FILE, sample_key)
        ours_x_np = np.transpose((patches_np / DEFAULT_MAX_PIXEL_VALUE).clip(0, 1), (0, 3, 1, 2)).astype(np.float32)

        if ours_x_np.shape[0] < 2:
            print(f"  [Ours] SKIP: only {ours_x_np.shape[0]} patch(es) — need at least 2.")
            continue
        ours_x = torch.from_numpy(ours_x_np).float().to(device)
        with torch.no_grad():
            ours_y = ours_model(ours_x).argmax(dim=1).cpu().numpy()
        print(f"  [Ours] {ours_x.shape[0]} patches, shape {ours_x.shape[1:]}, "
              f"y distribution: {np.bincount(ours_y, minlength=NUM_CLASSES)}")

        # Predictions from OARSI+KL multi-task model (Max_Multitask on KL head)
        with torch.no_grad():
            oarsi_kl_y = oarsi_kl_model(ours_x).argmax(dim=1).cpu().numpy()
        print(f"  [Ours (KL+OARSI)] y distribution: "
              f"{np.bincount(oarsi_kl_y, minlength=NUM_CLASSES)}")

        # Load Tiulpin patches
        print("  [Tiulpin] Loading 128x128 patch pair …")
        tiulpin_x_np = load_tiulpin_batch(TIULPIN_DATA_ROOT, sample_id, sample_side)
        tiulpin_x = torch.from_numpy(tiulpin_x_np).float().to(device)
        with torch.no_grad():
            tiulpin_y = tiulpin_model(tiulpin_x).argmax(dim=1).cpu().numpy()
        print(f"  [Tiulpin] {tiulpin_x.shape[0]} patches, shape {tiulpin_x.shape[1:]}, "
              f"y distribution: {np.bincount(tiulpin_y, minlength=NUM_CLASSES)}")

        # Per-sample configs
        sample_cfgs = {
            "Ours (CAM, KL)": {
                "model": ours_model,
                "x_batch": ours_x.cpu().numpy(),
                "y_batch": ours_y,
                "explain_func": ours_cam_explainer,
                "explain_kwargs": {"device": device},
            },
            "Ours (CAM, KL+OARSI)": {
                "model": oarsi_kl_model,
                "x_batch": ours_x.cpu().numpy(),
                "y_batch": oarsi_kl_y,
                "explain_func": ours_cam_explainer,
                "explain_kwargs": {"device": device},
            },
            "Tiulpin 2018 (GradCAM)": {
                "model": tiulpin_model,
                "x_batch": tiulpin_x.cpu().numpy(),
                "y_batch": tiulpin_y,
                "explain_func": tiulpin_gradcam_explainer,
                "explain_kwargs": {"device": device},
            },
        }

        # Run metrics for this sample
        for method_name, cfg in sample_cfgs.items():
            img_sz = OURS_IMG_SIZE if "Ours" in method_name else TIULPIN_IMG_SIZE
            metrics = _build_metrics(img_size=img_sz)
            for metric_name, metric_func in metrics.items():
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    scores = metric_func(
                        model=cfg["model"],
                        x_batch=cfg["x_batch"],
                        y_batch=cfg["y_batch"],
                        a_batch=None,
                        device=device,
                        explain_func=cfg["explain_func"],
                        explain_func_kwargs=cfg["explain_kwargs"],
                    )

                    # Flatten in case of nested/ragged per-instance results
                    if isinstance(scores, (list, tuple)):
                        pieces = []
                        for s in scores:
                            arr = np.asarray(s, dtype=float)
                            pieces.append(arr.reshape(-1))  # flatten any nested shape
                        flat = np.concatenate(pieces) if pieces else np.array([np.nan])
                    else:
                        flat = np.asarray(scores, dtype=float).reshape(-1)
                        if flat.size == 0:
                            flat = np.array([np.nan])


                    valid = flat[~np.isnan(flat)]
                    if valid.size == 0:
                        print(f"  [!] {method_name} / {metric_name}: all NaN, skipping this sample.")
                        val = np.nan
                    else:
                        val = float(np.mean(valid))

                    accum[method_name][metric_name].append(val)
                except Exception as e:
                    print(f"  [!] {method_name} / {metric_name} FAILED: {e}")
                    accum[method_name][metric_name].append(np.nan)

        gc.collect()
        torch.cuda.empty_cache()

    # 3. Aggregate: mean across samples
    results_agg = {}
    for method_name in method_names:
        results_agg[method_name] = {}
        for metric_name in _build_metrics():
            vals = accum[method_name][metric_name]
            if len(vals) > 0:
                results_agg[method_name][metric_name] = np.nanmean(vals)
            else:
                results_agg[method_name][metric_name] = np.nan

    print("\n\nAggregated results (mean across samples):")
    for method_name in method_names:
        print(f"  {method_name}:")
        for metric_name, val in results_agg[method_name].items():
            print(f"    {metric_name}: {val:.6f}")

    df = pd.DataFrame.from_dict(results_agg)
    df = df.T.abs()

    metric_order = ["Faithfulness", "Complexity", "Randomisation", "Robustness"]
    df = df[metric_order]

    # 4. Normalise & radar chart

    # Normalise: higher is better. Invert Robustness (lower AvgSensitivity = better).
    df_normalised = df.copy()
    for col in df_normalised.columns:
        if col != "Robustness":
            col_max = df_normalised[col].max()
            if col_max > 0 and np.isfinite(col_max):
                df_normalised[col] = df_normalised[col] / col_max
        else:
            col_min = df_normalised[col].replace(0, np.nan).min()
            if col_min > 0 and np.isfinite(col_min):
                df_normalised[col] = col_min / df_normalised[col].replace(0, np.nan)

    df_normalised_rank = df_normalised.rank()
    print("\nNormalised ranks:")
    print(df_normalised_rank)

    sns.set_theme(font_scale=1.8)
    plt.style.use("seaborn-v0_8-white")

    # teal = Ours (single-task KL), purple = Ours (multi-task KL+OARSI),
    # red = Tiulpin 2018
    colours_order = ["#008080", "#9467bd", "#d62728"]

    # Fill NaN ranks with the lowest rank so radar chart can still render
    df_safe_rank = df_normalised_rank.fillna(
        df_normalised_rank.values.max() + 1
        if np.isfinite(df_normalised_rank.values).any()
        else 1)

    data = [df_safe_rank.columns.values, (df_safe_rank.to_numpy())]
    theta = radar_factory(len(data[0]), frame="polygon")
    spoke_labels = data.pop(0)

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(projection="radar"))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    for i, (d, method) in enumerate(zip(data[0], method_names)):
        ax.plot(theta, d, label=method, color=colours_order[i], linewidth=5.0)
        ax.fill(theta, d, alpha=0.15)

    ax.set_varlabels(labels=["Faithfulness", "\nComplexity", "\nRandomisation", "Robustness"])
    ax.tick_params(axis="x", pad=25, labelsize=18)

    # Manually push out the horizontal-axis labels (Complexity, Robustness)
    # further than the vertical-axis ones
    for label, angle in zip(ax.get_xticklabels(), theta):
        x, y = label.get_position()
        if np.isclose(np.cos(angle), 0, atol=0.1):
            continue
        else:
            label.set_position((x, y))
            label.set_x(x + (0.08 if np.cos(angle) > 0 else -0.08))
            label.set_y(y - 0.06)

    rmax = np.nanmax(df_safe_rank.values) if np.isfinite(df_safe_rank.values).any() else 2
    ax.set_ylim(0, rmax + 1)
    ax.set_rgrids(np.arange(0, rmax + 0.5), labels=[])

    ax.set_title("", position=(0.5, 1.1), ha="center", fontsize=15,)

    ax.legend(
        loc="upper left", bbox_to_anchor=(0.95, 0.98),
        fontsize=11, handlelength=1.5, handletextpad=0.5, labelspacing=0.4, borderpad=0.6, frameon=False,)

    plt.tight_layout(rect=[0, 0, 0.95, 1])

    save_path = "quantus_cam_radar.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nRadar chart saved to: {os.path.abspath(save_path)}")
    plt.close(fig)

    print("\nDone.")
    return accum


if __name__ == "__main__":
    results = main()