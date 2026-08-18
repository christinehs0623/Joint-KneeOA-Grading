import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import pydicom
import torch.nn as nn
from captum.attr import Saliency, IntegratedGradients, GradientShap
import quant
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import Saliency
import h5py
from matplotlib.colors import LinearSegmentedColormap
import cv2
from model import PATCH_POINT_INDICES
# PATHS  (edit these)

# Each image filename should encode the patient+side so we can locate it.
TIULPIN_DATA_ROOT = "/data/net/datasets/OAI_Extracted/images/xray/preprocess_Knee_Radiographs_Tiulpin_png"          # PLACEHOLDER

# Our dataset: HDF5 file; groups are keyed like "9008884_R"
OURS_H5_FILE = "/usagers3/sigha/Joint-KneeOA-Grading/original_data/V00/V00_knee_patches_patient_grouped_16_100_all_feature.h5"               # PLACEHOLDER

# Saved model weights
TIULPIN_MODEL_PATH = "models/tiulpin_2018.pth"   # PLACEHOLDER
OURS_MODEL_PATH    = "models/ours_KL_quantus.pth"         # PLACEHOLDER

# SAMPLE TO EVALUATE
# Must exist in *both* datasets.  The HDF5 group key format is "<pid>_<side>".
SAMPLE_ID   = "9008884"   # patient ID
SAMPLE_SIDE = "R"         # "L" or "R"
SAMPLE_KEY  = f"{SAMPLE_ID}_{SAMPLE_SIDE}"   # HDF5 group name

# HARDWARE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# TIULPIN 2018

# 1a. Model definition

def ConvBlock3(inp, out, stride, pad):
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=3, stride=stride, padding=pad),
        nn.BatchNorm2d(out, eps=1e-3),
        nn.ReLU(inplace=True),
    )


class Branch(nn.Module):
    def __init__(self, bw):
        super().__init__()
        self.block1 = nn.Sequential(
            ConvBlock3(1, bw,     2, 0),
            ConvBlock3(bw, bw,    1, 0),
            ConvBlock3(bw, bw,    1, 0),
        )
        self.block2 = nn.Sequential(
            ConvBlock3(bw,     bw * 2, 1, 0),
            ConvBlock3(bw * 2, bw * 2, 1, 0),
        )
        self.block3 = ConvBlock3(bw * 2, bw * 4, 1, 0)

    def forward(self, x):
        o1 = F.max_pool2d(self.block1(x), 2)
        o2 = F.max_pool2d(self.block2(o1), 2)
        return F.avg_pool2d(self.block3(o2), 10).view(x.size(0), -1)


class KneeNet(nn.Module):
    """Siamese CNN - Tiulpin et al. 2018."""

    def __init__(self, bw=64, drop=0.2):
        super().__init__()
        self.branch = Branch(bw)
        self.final  = (
            nn.Sequential(nn.Dropout(p=drop), nn.Linear(2 * bw * 4, 5))
            if drop > 0
            else nn.Linear(2 * bw * 4, 5)
        )

    def forward(self, x1, x2):
        o1 = self.branch(x1)
        o2 = self.branch(x2)
        return self.final(torch.cat([o1, o2], dim=1))


# 1b. Image preprocessing (from tiulpin_2018/dataset_tiulpin2018.py)

def get_pair(pil_image):
    """
    Crop a left (medial) and right (lateral, flipped) 128x128 patch from
    a knee X-ray.  Reproduces the logic in dataset_tiulpin2018.py::get_pair().
    """
    s   = pil_image.size[0]          # width
    pad = int(np.floor(s / 3))
    ps  = 128
    l   = pil_image.crop([0,    pad, ps,   pad + ps])
    m   = pil_image.crop([s-ps, pad, s,    pad + ps])
    m   = m.transpose(Image.FLIP_LEFT_RIGHT)
    return l, m


def find_tiulpin_image(data_root, sample_id, sample_side, kl_label=None):
    """
    Locate the image file for *sample_id* in *data_root*.

    Strategy:
      - If kl_label is provided, only look in that class sub-folder.
      - Otherwise scan all class sub-folders (0-4).
      - The file stem must contain sample_id (and optionally sample_side).

    Returns the absolute filepath, or None if not found.
    """
    search_dirs = (
        [os.path.join(data_root, str(kl_label))]
        if kl_label is not None
        else [os.path.join(data_root, str(kl)) for kl in range(5)]
    )
    needle = sample_id  # at minimum the patient ID must appear in the filename
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            stem = os.path.splitext(fname)[0]
            if needle in stem:
                # Optionally also check side
                if sample_side and sample_side.upper() not in stem.upper():
                    continue
                return os.path.join(d, fname)
    return None


def tiulpin_predict(data_root, model_path, sample_id, sample_side, device):
    """
    Full inference pipeline for Tiulpin 2018.

    Returns
    -------
    kl_pred   : int   - predicted KL grade (0-4)
    kl_true   : int   - ground-truth KL grade inferred from folder name, or -1
    probs     : np.ndarray shape (5,) - softmax probabilities
    """
    #  locate image 
    img_path = find_tiulpin_image(data_root, sample_id, sample_side)
    if img_path is None:
        print(f"[Tiulpin] WARNING: image for '{sample_id}_{sample_side}' not found "
              f"under '{data_root}'. Using a blank 300x300 placeholder image.")
        img = Image.fromarray(np.zeros((300, 300), dtype=np.uint8))
        kl_true = -1
    else:
        print(f"[Tiulpin] Found image: {img_path}")
        img = Image.open(img_path).convert("L")
        # ground truth from parent folder name
        parent = os.path.basename(os.path.dirname(img_path))
        kl_true = int(parent) if parent.isdigit() else -1

    #  preprocessing 
    # Normalisation statistics.
    # PLACEHOLDER - replace with actual mean/std from your mean_std.npy file.
    mean = [0.5]
    std  = [0.5]

    patch_transform = T.Compose([
        T.ToTensor(),           # PIL → (C,H,W) float32 in [0,1]
        T.Lambda(lambda x: x.float()),
        T.Normalize(mean, std),
    ])

    l_patch, m_patch = get_pair(img)
    l_tensor = patch_transform(l_patch).unsqueeze(0).to(device)  # (1,1,128,128)
    m_tensor = patch_transform(m_patch).unsqueeze(0).to(device)

    # model 
    model = KneeNet(bw=64, drop=0.2).to(device)
    if os.path.isfile(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"[Tiulpin] Loaded weights from {model_path}")
    else:
        print(f"[Tiulpin] WARNING: model file not found at '{model_path}'. "
              "Running with random weights (placeholder).")

    model.eval()
    with torch.no_grad():
        logits = model(l_tensor, m_tensor)          # (1, 5)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
        kl_pred = int(probs.argmax())

    return kl_pred, kl_true, probs


# SECTION 2 - OUR MIL MODEL

# 2a. Minimal stub for the MIL model 
# The real model class lives in the main codebase (outside this repo).
# We attempt to import it; if unavailable we fall back to a placeholder stub
# that at least exercises the correct input/output contract.

try:
    # Adjust the import path to wherever your real model lives.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ours"))
    # from myutils import get_model   # PLACEHOLDER - import your real model builder
    def get_model(config):
        if config.model_type == "MIL": 
            from model import CompleteMILModel
            model = CompleteMILModel(config.FEATURE_EXTRACTOR_OUT_DIM,
                                        config.KL_NUM_CLASSES,
                                        config.AGGREGATION_TYPE).to(config.DEVICE)
        elif config.model_type == "MIL_ORG":
            from model import CompleteMILModel_ORG
            model = CompleteMILModel_ORG(config.FEATURE_EXTRACTOR_OUT_DIM,
                                        config.KL_NUM_CLASSES,
                                        config.AGGREGATION_TYPE).to(config.DEVICE)
        elif config.model_type == "MIL_MultiTask_imedslab":
            from model import CompleteMILModel_MultiTask_imedslab
            model = CompleteMILModel_MultiTask_imedslab(config.FEATURE_EXTRACTOR_OUT_DIM,
                                        config.OARSI_TASKS, config.AGGREGATION_TYPE).to(config.DEVICE)
        return model
    OUR_MODEL_AVAILABLE = True
    print("[Ours] Successfully imported model from 'ours/myutils.py'.")
except ImportError:
    OUR_MODEL_AVAILABLE = False
    print("[Ours] WARNING: could not import real MIL model. "
          "Using a placeholder stub that returns random logits.")

    class _PlaceholderMILModel(nn.Module):
        """
        Stand-in for the real MIL model.
        Accepts a list of patch-bag tensors, each shape (N_patches, C, H, W),
        and returns 5-class logits for the KL grade.
        Replace with the real model class when available.
        """
        def __init__(self, num_classes=5):
            super().__init__()
            # Tiny dummy network so the forward pass is well-defined.
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc   = nn.Linear(1, num_classes)   # 1-channel patches assumed

        def forward(self, bags):
            """bags: list of (N_patches, C, H, W) tensors."""
            bag = bags[0]                            # use first (only) bag
            # Average-pool every patch, then mean across patches -> (C,)
            pooled = self.pool(bag).squeeze(-1).squeeze(-1)   # (N, C)
            agg    = pooled.mean(dim=0, keepdim=True)          # (1, C)
            return self.fc(agg)                                # (1, num_classes)


def build_our_model(model_path, device):
    """Instantiate and load the MIL model."""
    if OUR_MODEL_AVAILABLE:
        # Build config-free version of our model.
        # PLACEHOLDER - replace the class/args with those of your real model.
        class _MinimalConfig:
            model_type      = "MIL_MultiTask_imedslab"
            feedback_type   = "off"
            multitask_type  = "off"
            KL_NUM_CLASSES     = 5
            FEATURE_EXTRACTOR_OUT_DIM = 128
            AGGREGATION_TYPE = "attention"
            DEVICE = device
            OARSI_TASKS  = {
                "kl":   5,   # head_0 — KL grade, 5 classes (0-4)
                "jsnm": 4,   # head_1
                "jsnl": 4,   # head_2
                "osfm": 4,   # head_3
                "ostm": 4,   # head_4
                "ostl": 4,   # head_5
                "osfl": 4,   # head_6
            }
            
            # Add any other attributes your get_model() needs here.
        cfg   = _MinimalConfig()
        model = get_model(cfg)
    else:
        model = _PlaceholderMILModel(num_classes=5)

    print("model_type:", cfg.model_type)
    print("FEATURE_EXTRACTOR_OUT_DIM:", getattr(cfg, "FEATURE_EXTRACTOR_OUT_DIM", "MISSING"))
    if os.path.isfile(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"[Ours] Loaded weights from {model_path}")
    else:
        print(f"[Ours] WARNING: model file not found at '{model_path}'. "
              "Running with random/uninitialised weights (placeholder).")

    return model.to(device)


# 2b. Dataset loading (from ours/dataset.py) 

def load_our_sample(h5_file, sample_key):
    """
    Load patches and label for *sample_key* from the HDF5 file.

    Returns
    -------
    patches_np : np.ndarray (N, H, W, C) float32, raw pixel values
    kl_true    : int
    """

    if not os.path.isfile(h5_file):
        print(f"[Ours] WARNING: HDF5 file not found at '{h5_file}'. "
              "Using a random placeholder bag (4 patches, 1x16x16).")
        patches_np = np.random.rand(4, 16, 16, 1).astype(np.float32)
        return patches_np, -1

    with h5py.File(h5_file, "r") as hf:
        if sample_key not in hf:
            raise KeyError(f"Sample '{sample_key}' not found in {h5_file}. "
                           f"Available keys (first 10): {list(hf.keys())[:10]}")
        grp       = hf[sample_key]
        patches   = grp["patches"][:]          # (N, H, W, C) float32
        kl_true   = int(grp["kl_grade"][0])

    return patches, kl_true


def our_predict(h5_file, model_path, sample_key, device):
    """
    Full inference pipeline for our MIL model.

    Returns
    -------
    kl_pred  : int
    kl_true  : int   (-1 if unknown)
    probs    : np.ndarray (5,)
    """
    # data 
    patches_np, kl_true = load_our_sample(h5_file, sample_key)

    # Normalisation statistics.
    # PLACEHOLDER - replace with the actual mean/std saved during training.
    DEFAULT_MAX_PIXEL_VALUE = 65535.0
    mean = 0.5
    std  = 0.5

    # Build normalisation transform (operates on (C,H,W) float tensor in [0,1])
    norm_transform = T.Normalize(mean=[mean], std=[std])

    # Convert raw patches to tensors
    processed = []
    for i in range(patches_np.shape[0]):
        patch = patches_np[i]                                  # (H, W, C)
        patch = (patch / DEFAULT_MAX_PIXEL_VALUE).clip(0, 1)   # [0,1]
        patch = np.transpose(patch, (2, 0, 1))                 # (C,H,W)
        t     = torch.from_numpy(patch).float()
        t     = norm_transform(t)
        processed.append(t)

    patch_bag = torch.stack(processed, dim=0).to(device)       # (N, C, H, W)

    # model
    model = build_our_model(model_path, device)
    model.eval()
    with torch.no_grad():
        # logits = model([patch_bag])      # list of one bag → (1, 5)
        oai_preds_all, att_scores, patch_embeddings, agg_features = model([patch_bag])
        logits = oai_preds_all["kl"]
        # Handle both plain tensor output and dict output (multitask)
        if isinstance(logits, dict):
            logits = logits["kl"]
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
        kl_pred = int(probs.argmax())

    return kl_pred, kl_true, probs


# SECTION 3 - QUANTUS XAI EVALUATION
#
# Adds explainability evaluation on top of the models above using Quantus:
#   https://github.com/understandable-machine-intelligence-lab/quantus
#
# We evaluate four metric categories (one representative metric each):
#   - Faithfulness   : quantus.FaithfulnessCorrelation
#   - Robustness     : quantus.AvgSensitivity
#   - Complexity     : quantus.Sparseness
#   - Randomisation  : quantus.RandomLogit
#
# Attributions are produced with Captum (Saliency, Integrated Gradients,
# Gradient SHAP) to mirror the reference figure, and the aggregated scores are
# drawn on a radar / spider chart.
#
# Install dependencies with:
#   pip install quantus captum matplotlib
# 

# 3a. Model wrappers 
# Quantus & Captum expect a standard classifier:  forward(x) -> (B, num_classes)
# Neither of our models exposes that contract directly, so we wrap them.


class TiulpinWrapper(nn.Module):
    """
    Adapts the Siamese KneeNet to a single-tensor classifier.

    Quantus perturbs one input tensor at a time, so we feed the *same* patch
    into both Siamese branches.  Input:  (B, 1, 128, 128) -> (B, 5).
    """

    def __init__(self, kneenet: "KneeNet"):
        super().__init__()
        self.kneenet = kneenet

    def forward(self, x):
        return self.kneenet(x, x)


class MILWrapper(nn.Module):
    """
    Adapts the MIL model to a per-patch classifier.

    The MIL model normally consumes a *bag* of patches and returns one label
    for the whole bag.  For pixel-level XAI evaluation we treat every patch as
    an independent sample: input (B, C, H, W) -> logits (B, 5).

    We go directly through patch_feature_extractor → KL head, bypassing the
    complex MIL aggregator and dict-output machinery.  For single-patch bags
    this is equivalent to the full pipeline (attention weight = 1.0).
    """

    def __init__(self, mil_model: nn.Module, num_classes: int = 5):
        super().__init__()
        self.mil_model = mil_model
        self.num_classes = num_classes

    def forward(self, x):
        # x: (B, C, H, W)
        # Extract embeddings for all patches at once (batched)
        embeddings = self.mil_model.patch_feature_extractor(x)  # (B, out_dim)
        # Apply only the KL classification head
        logits = self.mil_model.oai_heads.head_0(embeddings)  # (B, 5)
        return logits

    def predict(self, x):
        """
        Quantus-compatible predict method that returns numpy array.
        quantus wraps models with PyTorchModel which may call model.predict().
        """
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def shape_input(self, x, shape, channel_first=True):
        """
        Quantus-compatible shape_input method required by PyTorchModel.
        """
        return x


# 3b. Attribution generation (Captum)

def generate_attributions(model, x_batch, y_batch, device):
    """
    Produce attribution maps for several XAI methods using Captum.

    Parameters
    ----------
    model    : nn.Module      - wrapped classifier, forward(x) -> (B, num_classes)
    x_batch  : np.ndarray     - (B, C, H, W)
    y_batch  : np.ndarray     - (B,) int labels
    device   : torch.device

    Returns
    -------
    dict  { method_name : attribution_np (B, C, H, W) }
    """
   

    x = torch.tensor(x_batch, dtype=torch.float32, device=device)
    x.requires_grad_(True)
    y = torch.tensor(y_batch, dtype=torch.long, device=device)

    model.eval()

    attributions = {}

    # Saliency (SA)
    sa = Saliency(model)
    attributions["Saliency (SA)"] = (
        sa.attribute(x, target=y).detach().cpu().numpy()
    )

    # Integrated Gradients (IG)
    ig = IntegratedGradients(model)
    attributions["Integrated Gradients (IG)"] = (ig.attribute(x, target=y, n_steps=20).detach().cpu().numpy())

    # Gradient SHAP (GS)
    gs = GradientShap(model)
    baselines = torch.cat([x * 0, x * 0 + x.mean()], dim=0)
    attributions["Gradient Shap (GS)"] = (
        gs.attribute(x, baselines=baselines, target=y).detach().cpu().numpy())

    return attributions


# 3b-bis. Custom Faithfulness Correlation

def faithfulness_correlation(model, x_batch, y_batch, a_batch, device, nr_runs=10, subset_size=None, baseline=0.0):
    """
    Self-contained Faithfulness-Correlation metric.

    Mirrors ``quantus.FaithfulnessCorrelation`` but calls *our own* model
    forward, which reliably returns a ``(B, num_classes)`` tensor.  Quantus's
    built-in metric routes predictions through ``PyTorchModel.predict`` and then
    does ``float(model.predict(x)[:, y])`` internally - with our wrapped models
    (the per-patch looping ``MILWrapper`` and the Siamese ``TiulpinWrapper``)
    that path can yield an array instead of a scalar, raising
    ``"only 0-dimensional arrays can be converted to Python scalars"``.
    Computing the metric ourselves avoids that fragile code path entirely.

    Semantics are identical to FaithfulnessCorrelation:
      - Take the model's probability for the target class as a baseline.
      - Repeatedly mask a random subset of features to a baseline value,
        re-predict, and record (a) the drop in target probability and
        (b) the summed attribution over the masked subset.
      - Pearson-correlate the attribution sums with the probability drops for
        each sample, then average across the batch.  Higher = more faithful.
    """
    model = model.to(device).eval()

    x_t = torch.tensor(x_batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        base_probs = F.softmax(model(x_t), dim=1).cpu().numpy()   # (B, num_classes)

    B = x_batch.shape[0]
    n_features = int(np.prod(x_batch.shape[1:]))
    if subset_size is None:
        subset_size = max(4, x_batch.shape[-1] // 4)
    subset_size = int(min(subset_size, n_features))

    per_sample = []
    for b in range(B):
        y = int(y_batch[b])
        base_pred = float(base_probs[b, y])
        a_flat = np.asarray(a_batch[b], dtype=float).flatten()

        pred_deltas, att_sums = [], []
        for _ in range(nr_runs):
            idx = np.random.choice(n_features, subset_size, replace=False)

            x_pert = x_batch[b].copy().reshape(-1)
            x_pert[idx] = baseline
            x_pert = x_pert.reshape(x_batch.shape[1:])

            with torch.no_grad():
                xt = torch.tensor(x_pert[None], dtype=torch.float32, device=device)
                p = float(F.softmax(model(xt), dim=1).cpu().numpy()[0, y])

            pred_deltas.append(base_pred - p)
            att_sums.append(float(np.sum(a_flat[idx])))

        pd  = np.asarray(pred_deltas, dtype=float)
        ats = np.asarray(att_sums,   dtype=float)
        if np.std(pd) < 1e-12 or np.std(ats) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(ats, pd)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        per_sample.append(corr)

    return float(np.mean(per_sample)) if per_sample else float("nan")


# 3c. Quantus metric evaluation 

def evaluate_xai(model, x_batch, y_batch, device):
    """
    Run the four Quantus metric categories for every attribution method.


    Returns
    -------
    dict  { method_name : { "Faithfulness": v, "Robustness": v,
                            "Complexity": v, "Randomisation": v } }
    """


    model = model.to(device).eval()

    # Generate attributions once per method.
    attributions = generate_attributions(model, x_batch, y_batch, device)

    # A custom explain_func so Quantus can (re)generate Saliency maps internally
    # for metrics that need to re-explain perturbed inputs (e.g. robustness).
    def explain_func(model, inputs, targets, **kwargs):
        
        model_t = model
        x = torch.tensor(inputs, dtype=torch.float32, device=device)
        x.requires_grad_(True)
        t = torch.tensor(targets, dtype=torch.long, device=device)
        a = Saliency(model_t).attribute(x, target=t)
        return a.detach().cpu().numpy()

    # Faithfulness is computed with our own robust helper (see
    # faithfulness_correlation) instead of quantus.FaithfulnessCorrelation, whose
    # internal predict path raised
    # "only 0-dimensional arrays can be converted to Python scalars"
    # with our wrapped models.  The remaining three categories use Quantus.
    metrics = {
        "Robustness": quant.AvgSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            abs=True,

            return_aggregate=True,
            disable_warnings=True,
        ),
        "Complexity": quant.Sparseness(
            abs=True,
            return_aggregate=True,
            disable_warnings=True,
        ),
        "Randomisation": quant.RandomLogit(
            num_classes=5,
            similarity_func=quant.similarity_func.ssim,
            return_aggregate=True,
            disable_warnings=True,
        ),
    }

    results = {}
    for method_name, a_batch in attributions.items():
        print(f"\n[Quantus] Evaluating '{method_name}' …")
        method_scores = {}

        # Faithfulness via our robust self-contained implementation.
        try:
            fc = faithfulness_correlation(
                model, x_batch, y_batch, a_batch, device,
                nr_runs=10, subset_size=max(4, x_batch.shape[-1] // 4),
            )
        except Exception as e:
            print(f"  [!] Faithfulness failed: {e}")
            fc = np.nan
        method_scores["Faithfulness"] = fc
        print(f"  {'Faithfulness':14s}: {fc:.4f}")

        for metric_name, metric in metrics.items():

            try:
                score = metric(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=str(device),
                    explain_func=explain_func,
                )
                val = float(np.mean(np.asarray(score, dtype=float)))
            except Exception as e:  # keep the pipeline running end-to-end
                print(f"  [!] {metric_name} failed: {e}")
                val = np.nan
            method_scores[metric_name] = val
            print(f"  {metric_name:14s}: {val:.4f}")
        results[method_name] = method_scores

    return results


# 3d. Radar / spider chart

def plot_radar(results, save_path="quantus_radar.png", title="Quantus XAI Evaluation"):

    """
    Draw a radar chart with one polygon per attribution method.

    Axes: Faithfulness, Robustness, Complexity, Randomisation.
    Scores are min-max normalised per axis so polygons are comparable.
    """

    axes_labels = ["Faithfulness", "Robustness", "Complexity", "Randomisation"]

    # For robustness, lower AvgSensitivity is better -> invert before scaling.
    processed = {m: dict(s) for m, s in results.items()}
    for m in processed:
        r = processed[m].get("Robustness", np.nan)
        if not np.isnan(r):
            processed[m]["Robustness"] = -r

    # Min-max normalise each axis across methods.
    normed = {m: {} for m in processed}
    for ax in axes_labels:
        vals = np.array([processed[m].get(ax, np.nan) for m in processed], float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            for m in processed:
                normed[m][ax] = 0.0
            continue
        vmin, vmax = finite.min(), finite.max()
        span = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
        for m, v in zip(processed.keys(), vals):
            normed[m][ax] = 0.5 if not np.isfinite(v) else (v - vmin) / span

    # Angles for each axis (close the loop).
    n = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    colors = {
        "Saliency (SA)":             "#2a7f7f",
        "Integrated Gradients (IG)": "#f0a000",
        "Gradient Shap (GS)":        "#1f3b73",
        "FusionGrad (FG)":           "#d62728",
    }

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=13)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)

    for method, scores in normed.items():
        vals = [scores[ax_name] for ax_name in axes_labels]
        vals += vals[:1]
        c = colors.get(method, None)
        ax.plot(angles, vals, linewidth=2.5, label=method, color=c)
        ax.fill(angles, vals, alpha=0.12, color=c)

    ax.set_title(title, fontsize=15, pad=20)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Quantus] Radar chart saved to: {os.path.abspath(save_path)}")
    plt.close(fig)


# 3e. Batch builders + per-model Quantus runner

def find_common_samples(h5_file, tiulpin_data_root):
    """
    Discover all (sample_id, sample_side, sample_key) tuples that exist in
    both the HDF5 file and the Tiulpin PNG dataset.

    Returns a list of tuples sorted by sample_id then side.
    """
    

    common = []
    if not os.path.isfile(h5_file):
        print(f"[find_common_samples] WARNING: HDF5 file not found at '{h5_file}'.")
        return common

    with h5py.File(h5_file, "r") as hf:
        all_keys = list(hf.keys())

    for key in all_keys:
        # Keys are like "9008884_R" or "9008884_L"
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        sample_id, sample_side = parts
        if sample_side.upper() not in ("L", "R"):
            continue
        img_path = find_tiulpin_image(tiulpin_data_root, sample_id, sample_side.upper())
        if img_path is not None:
            common.append((sample_id, sample_side.upper(), key))

    # Sort by patient ID (numeric) then side
    def _sort_key(tup):
        sid = tup[0]
        try:
            sid_num = int(sid)
        except ValueError:
            sid_num = 0
        return (sid_num, tup[1])

    common.sort(key=_sort_key)
    print(f"[find_common_samples] Found {len(common)} common samples.")
    return common


def load_tiulpin_batch(data_root, sample_id, sample_side):
    """
    Build an (N, 1, 128, 128) float32 batch from the medial/lateral patch pair
    of the Tiulpin sample, normalised exactly as in tiulpin_predict.
    """
    img_path = find_tiulpin_image(data_root, sample_id, sample_side)
    if img_path is None:
        print(f"[Tiulpin] WARNING: image for '{sample_id}_{sample_side}' not found; "
              "using a blank 300x300 placeholder for XAI.")
        img = Image.fromarray(np.zeros((300, 300), dtype=np.uint8))
    else:
        img = Image.open(img_path).convert("L")

    patch_transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x.float()),
        T.Normalize([0.5], [0.5]),
    ])
    l_patch, m_patch = get_pair(img)
    l_t = patch_transform(l_patch)          # (1, 128, 128)
    m_t = patch_transform(m_patch)
    x_batch = torch.stack([l_t, m_t], dim=0).numpy().astype(np.float32)  # (2,1,128,128)
    return x_batch


def run_quantus_for_model(name, wrapped_model, x_batch, device, radar_path):
    """
    Run the four XAI metric categories for one wrapped model, print a table and
    save a radar chart.  Returns the results dict.
    """
    wrapped_model = wrapped_model.to(device).eval()

    # Target label per sample = the model's own predicted class.
    with torch.no_grad():
        logits  = wrapped_model(torch.tensor(x_batch, device=device))
        y_batch = logits.argmax(dim=1).cpu().numpy().astype(int)

    print(f"\n[{name}] x_batch: {x_batch.shape}, y_batch: {y_batch.shape}")
    results = evaluate_xai(wrapped_model, x_batch, y_batch, device)
    plot_radar(results, save_path=radar_path,
               title=f"Quantus XAI Evaluation - {name}")
    return results


# SECTION 4 - SALIENCY OVERLAY ON ORIGINAL DICOM
#
# Projects the Captum "Saliency" attribution maps (computed during the Quantus
# XAI evaluation above) back onto the *original* full-resolution bilateral
# knee X-ray (.dcm), for both models:
#
#   - Ours (MIL):    each of the per-patch saliency maps is pasted back at its
#                     corresponding landmark location (PATCH_POINT_INDICES).
#   - Tiulpin 2018:   the medial/lateral 128x128 saliency maps are pasted back
#                      at the crop offsets used to build the Tiulpin patch
#                      pair, where the crop itself is taken from the *same*
#                      bilateral .dcm using the landmark bounding box for the
#                      target knee side (see get_pair()).
#
# Paths (edit when the real files are available):
DICOM_DIR     = "sample_data/OAI" # PLACEHOLDER
LANDMARKS_NPZ = "./id_shapes_LR_V00.npz"                                        # PLACEHOLDER


def process_xray_image(img, cut_min=5, cut_max=99, multiplier=255):
    """
    Global contrast normalisation (Tiulpin et al. 2018 style).
    Clips to [cut_min, cut_max] percentiles, then rescales to [0, multiplier].
    """
    img = img.copy().astype(np.float64)
    lim1, lim2 = np.percentile(img, [cut_min, cut_max])
    img[img < lim1] = lim1
    img[img > lim2] = lim2
    img -= lim1
    denom = img.max()
    if denom > 0:
        img /= denom
    img *= multiplier
    return img


def load_dicom_image(patient_id, dicom_dir=DICOM_DIR):
    """
    Load and contrast-normalise the full bilateral DICOM for *patient_id*.
    Returns a 2D float array, or a blank placeholder image if not found.
    """
    fpath = os.path.join(dicom_dir, f"{patient_id}.dcm")
    if not os.path.isfile(fpath):
        print(f"[Saliency] WARNING: DICOM not found at '{fpath}'. "
              "Using a blank 4320x3560 placeholder image.")
        return np.zeros((4320, 3560), dtype=np.float64)

    
    dicom_data = pydicom.dcmread(fpath)
    image = dicom_data.pixel_array.astype(np.float64)
    image = process_xray_image(image, 5, 99, 65535)
    image = image.reshape((dicom_data.Rows, dicom_data.Columns))
    return image


def load_landmarks(patient_id, npz_path=LANDMARKS_NPZ):
    """
    Returns (shapes_L, shapes_R) landmark arrays (each (N_pts, 2)) for
    *patient_id*, or (None, None) if the landmarks file/patient is missing.
    """
    if not os.path.isfile(npz_path):
        print(f"[Saliency] WARNING: landmarks file not found at '{npz_path}'.")
        return None, None

    data = np.load(npz_path)
    patient_ids = data["id"]
    matches = np.where(np.array(patient_ids) == patient_id)[0]
    if matches.size == 0:
        print(f"[Saliency] WARNING: patient '{patient_id}' not found in landmarks file.")
        return None, None

    idx = matches.item()
    return data["shapes_L"][idx], data["shapes_R"][idx]


def compute_patch_half_size(image_shape, ref_patch_dim=100, ref_img_area=3560 * 4320):
    """Half-size (in px) of a landmark patch box, scaled to the image resolution."""
    img_area = image_shape[0] * image_shape[1]
    return (ref_patch_dim / np.sqrt(ref_img_area)) * np.sqrt(img_area) / 2.0


def patch_from_point(point, half_size):
    """Return (topLeft, botRight) pixel coords of a square box centred on *point*."""
    top_left = (int(point[0] - half_size), int(point[1] - half_size))
    bot_right = (int(point[0] + half_size), int(point[1] + half_size))
    return top_left, bot_right


def create_reds_alpha_cmap():
    """Reds colormap where low values are fully transparent (for overlays)."""
    
    ncolors = 256
    base = plt.get_cmap("Reds")(np.arange(ncolors))
    v = np.arange(ncolors) / (ncolors - 1)
    alpha = np.where(v >= 0.5, 1.0, 2 * v)
    base[:, -1] = alpha
    return LinearSegmentedColormap.from_list("Reds_alpha", base)


def _resize_map(arr, size_hw):
    """Resize a 2D array to (H, W) = size_hw, using cv2 if available else PIL."""
    h, w = size_hw
    if h <= 0 or w <= 0:
        return None
    try:
        
        return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        img = Image.fromarray(arr.astype(np.float32), mode="F")
        img = img.resize((w, h), Image.BILINEAR)
        return np.array(img)


def _smooth(canvas, sigma):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(canvas, sigma=sigma)
    except ImportError:
        return canvas


def _to_2d_map(sal_map):
    """Collapse a (C,H,W) or (H,W) attribution map to a single 2D array via mean|abs|."""
    sal_map = np.asarray(sal_map)
    if sal_map.ndim == 3:
        return np.mean(np.abs(sal_map), axis=0)
    return np.abs(sal_map)


def _save_overlay(base_img, canvas, cmap, alpha, save_path, title, bbox=None):
    """Blend `canvas` (already min-max normalised) over `base_img` and save to disk."""
   
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.imshow(base_img, cmap="gray")
    if canvas is not None:
        ax.imshow(canvas, cmap=cmap, alpha=alpha)
    if bbox is not None:
        (x0, y0), (x1, y1) = bbox
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    edgecolor="cyan", facecolor="none", linewidth=2))
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saliency] Saved: {os.path.abspath(save_path)}")


def visualize_saliency_ours_on_dicom(patient_id, side, saliency_maps,
                                      save_path="saliency_ours_on_dicom.png"):
    """
    Paste per-patch Saliency attribution maps back onto the original bilateral
    DICOM, at the landmark locations used to extract 'ours' patches.

    Parameters
    ----------
    patient_id    : str
    side          : "L" or "R"
    saliency_maps : np.ndarray (N_patches, C, H, W) - Captum Saliency output for
                    the MIL model (one map per patch, in the same order as
                    PATCH_POINT_INDICES / the HDF5 patch bag for this sample).
    """
    

    base_img = load_dicom_image(patient_id, DICOM_DIR)
    shapes_L, shapes_R = load_landmarks(patient_id, LANDMARKS_NPZ)

    if shapes_L is None:
        print("[Saliency-Ours] Landmarks unavailable; saving raw DICOM only.")
        _save_overlay(base_img, None, None, 0, save_path,
                      f"Ours - {patient_id}_{side} (no landmarks)")
        return

    shapes = shapes_L if side.upper() == "L" else shapes_R
    half_size = compute_patch_half_size(base_img.shape)
    patch_dim = max(1, int(round(half_size * 2)))

    canvas = np.zeros_like(base_img, dtype=np.float64)
    n_pts = min(saliency_maps.shape[0], len(PATCH_POINT_INDICES))

    for i in range(n_pts):
        pt_idx = PATCH_POINT_INDICES[i]
        if pt_idx >= shapes.shape[0]:
            continue
        center = shapes[pt_idx]
        sal_map = _to_2d_map(saliency_maps[i])

        top_left, bot_right = patch_from_point(center, half_size)
        x0, y0 = top_left
        x1, y1 = bot_right

        x0_clip, x1_clip = max(0, x0), min(base_img.shape[1], x1)
        y0_clip, y1_clip = max(0, y0), min(base_img.shape[0], y1)
        if x1_clip <= x0_clip or y1_clip <= y0_clip:
            continue

        resized = _resize_map(sal_map, (y1 - y0, x1 - x0))
        if resized is None:
            continue
        h_slice = resized[(y0_clip - y0):(y1_clip - y0), (x0_clip - x0):(x1_clip - x0)]
        canvas[y0_clip:y1_clip, x0_clip:x1_clip] += h_slice

    c_min, c_max = canvas.min(), canvas.max()
    canvas_norm = (canvas - c_min) / (c_max - c_min) if c_max > c_min else canvas
    canvas_norm = _smooth(canvas_norm, sigma=max(1.0, (patch_dim - 1) / 8))

    cmap = create_reds_alpha_cmap()
    _save_overlay(base_img, canvas_norm, cmap, 0.8, save_path,
                  f"Ours - Saliency overlay - {patient_id}_{side}")


def visualize_saliency_tiulpin_on_dicom(patient_id, side, saliency_l, saliency_m,
                                         save_path="saliency_tiulpin_on_dicom.png"):
    """
    Paste the Tiulpin medial/lateral 128x128 Saliency maps back onto the
    original bilateral DICOM.

    The region Tiulpin "sees" is obtained by first cropping the full DICOM to
    the target knee side using the landmark bounding box (with padding), then
    reproducing get_pair()'s geometry on that crop, exactly as
    load_tiulpin_batch()/tiulpin_predict() would if fed this crop. The
    saliency maps are pasted back at crop offset + patch offset.

    Parameters
    ----------
    saliency_l : np.ndarray (C,H,W) or (H,W) - Saliency map for the lateral patch (`l`).
    saliency_m : np.ndarray (C,H,W) or (H,W) - Saliency map for the medial patch (`m`),
                 as produced directly by the model on the flipped patch; this
                 function un-flips it back before pasting.
    """
    base_img = load_dicom_image(patient_id, DICOM_DIR)
    shapes_L, shapes_R = load_landmarks(patient_id, LANDMARKS_NPZ)

    if shapes_L is None:
        print("[Saliency-Tiulpin] Landmarks unavailable; saving raw DICOM only.")
        _save_overlay(base_img, None, None, 0, save_path,
                      f"Tiulpin 2018 - {patient_id}_{side} (no landmarks)")
        return

    shapes = shapes_L if side.upper() == "L" else shapes_R

    # 1. Crop to the target knee side using the landmark bounding box
    padding = 200  # ~2 patch-widths of context, matches ecam-style padding
    min_xy = shapes.min(axis=0)
    max_xy = shapes.max(axis=0)
    cx0 = max(0, int(min_xy[0] - padding))
    cy0 = max(0, int(min_xy[1] - padding))
    cx1 = min(base_img.shape[1], int(max_xy[0] + padding))
    cy1 = min(base_img.shape[0], int(max_xy[1] + padding))

    crop = base_img[cy0:cy1, cx0:cx1]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        print("[Saliency-Tiulpin] WARNING: empty crop region; saving raw DICOM only.")
        _save_overlay(base_img, None, None, 0, save_path,
                      f"Tiulpin 2018 - {patient_id}_{side} (empty crop)")
        return

    # 2. Reproduce get_pair() geometry on the crop
    s = crop.shape[1]              # crop width
    pad = int(np.floor(s / 3))
    ps = 128
    l_box = (0, pad, ps, pad + ps)            # lateral patch box within crop
    m_box = (s - ps, pad, s, pad + ps)        # medial patch box within crop (pre-flip)

    canvas = np.zeros_like(base_img, dtype=np.float64)

    def _paste(sal_map, box, flip):
        sal_map = _to_2d_map(sal_map)
        if flip:
            sal_map = np.fliplr(sal_map)
        x0b, y0b, x1b, y1b = box
        ax0, ay0 = cx0 + x0b, cy0 + y0b
        ax1, ay1 = cx0 + x1b, cy0 + y1b
        ax0c, ax1c = max(0, ax0), min(base_img.shape[1], ax1)
        ay0c, ay1c = max(0, ay0), min(base_img.shape[0], ay1)
        if ax1c <= ax0c or ay1c <= ay0c:
            return
        resized = _resize_map(sal_map, (ay1 - ay0, ax1 - ax0))
        if resized is None:
            return
        h_slice = resized[(ay0c - ay0):(ay1c - ay0), (ax0c - ax0):(ax1c - ax0)]
        canvas[ay0c:ay1c, ax0c:ax1c] += h_slice

    _paste(saliency_l, l_box, flip=False)
    _paste(saliency_m, m_box, flip=True)   # medial patch was flipped by get_pair()

    c_min, c_max = canvas.min(), canvas.max()
    canvas_norm = (canvas - c_min) / (c_max - c_min) if c_max > c_min else canvas
    canvas_norm = _smooth(canvas_norm, sigma=8)

    cmap = create_reds_alpha_cmap()
    bbox = ((cx0, cy0), (cx1, cy1))
    _save_overlay(base_img, canvas_norm, cmap, 0.8, save_path,
                  f"Tiulpin 2018 - Saliency overlay - {patient_id}_{side}", bbox=bbox)


# SECTION 5 - COMPARISON
def main():


    print("=" * 60)
    print("KL Grade Inference Comparison")
    print(f"  Sample : {SAMPLE_KEY}")
    print("=" * 60)

    # Tiulpin 2018 
    print("\n[1/2] Running Tiulpin 2018 model …")
    t_pred, t_true, t_probs = tiulpin_predict(
        data_root   = TIULPIN_DATA_ROOT,
        model_path  = TIULPIN_MODEL_PATH,
        sample_id   = SAMPLE_ID,
        sample_side = SAMPLE_SIDE,
        device      = DEVICE,
    )

    # Our model
    print("\n[2/2] Running our MIL model …")
    o_pred, o_true, o_probs = our_predict(
        h5_file    = OURS_H5_FILE,
        model_path = OURS_MODEL_PATH,
        sample_key = SAMPLE_KEY,
        device     = DEVICE,
    )

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'':30s} {'Tiulpin 2018':>15s} {'Ours':>10s}")
    print(f"{'Predicted KL grade':30s} {t_pred:>15d} {o_pred:>10d}")
    print(f"{'Ground-truth KL grade':30s} {str(t_true):>15s} {str(o_true):>10s}")
    print()
    print("Tiulpin 2018 - class probabilities:")
    for kl, p in enumerate(t_probs):
        print(f"  KL {kl}: {p:.4f}")
    print()
    print("Our model - class probabilities:")
    for kl, p in enumerate(o_probs):
        print(f"  KL {kl}: {p:.4f}")
    print("=" * 60)

    # Quantus XAI evaluation + radar chart 

    print("\n" + "=" * 60)
    print("QUANTUS XAI EVALUATION")
    print("=" * 60)

    # Tiulpin 2018 model 
    print("\n[Tiulpin 2018] Building wrapped model for XAI …")
    kneenet = KneeNet(bw=64, drop=0.2).to(DEVICE)
    if os.path.isfile(TIULPIN_MODEL_PATH):
        kneenet.load_state_dict(torch.load(TIULPIN_MODEL_PATH, map_location=DEVICE))
        print(f"[Tiulpin] Loaded weights from {TIULPIN_MODEL_PATH}")
    else:
        print(f"[Tiulpin] WARNING: model file not found at '{TIULPIN_MODEL_PATH}'. "
              "Running with random weights (placeholder).")
    tiulpin_wrapped = TiulpinWrapper(kneenet).to(DEVICE)
    tiulpin_x = load_tiulpin_batch(TIULPIN_DATA_ROOT, SAMPLE_ID, SAMPLE_SIDE)

    # Our MIL model 
    print("\n[Ours] Building wrapped model for XAI …")
    mil_model    = build_our_model(OURS_MODEL_PATH, DEVICE)
    ours_wrapped = MILWrapper(mil_model, num_classes=5).to(DEVICE)
    patches_np, _ = load_our_sample(OURS_H5_FILE, SAMPLE_KEY)
    DEFAULT_MAX_PIXEL_VALUE = 65535.0
    ours_x = np.transpose(
        (patches_np / DEFAULT_MAX_PIXEL_VALUE).clip(0, 1), (0, 3, 1, 2)
    ).astype(np.float32)                                    # (N, C, H, W)

    #  Run the XAI evaluation for BOTH models 
    xai_results = {}
    xai_results["Tiulpin 2018"] = run_quantus_for_model(
        "Tiulpin 2018", tiulpin_wrapped, tiulpin_x, DEVICE, "quantus_radar_tiulpin.png"
    )
    xai_results["Ours"] = run_quantus_for_model(
        "Ours", ours_wrapped, ours_x, DEVICE, "quantus_radar_ours.png"
    )

    #  Saliency overlay on the original DICOM (both models) 
    print("\n" + "=" * 60)
    print("SALIENCY OVERLAY ON ORIGINAL DICOM")
    print("=" * 60)

    saliency_paths = {}

    #  Tiulpin 2018 : Saliency maps for the [lateral, medial] pair 
    try:
        with torch.no_grad():
            t_logits = tiulpin_wrapped(torch.tensor(tiulpin_x, device=DEVICE))
            t_y = t_logits.argmax(dim=1).cpu().numpy().astype(int)
        t_attr = generate_attributions(tiulpin_wrapped, tiulpin_x, t_y, DEVICE)
        t_saliency = t_attr["Saliency (SA)"]   # (2, 1, 128, 128) -> [l, m]

        tiulpin_saliency_path = "saliency_tiulpin_on_dicom.png"
        visualize_saliency_tiulpin_on_dicom(
            patient_id  = SAMPLE_ID,
            side        = SAMPLE_SIDE,
            saliency_l  = t_saliency[0],
            saliency_m  = t_saliency[1],
            save_path   = tiulpin_saliency_path,
        )
        saliency_paths["tiulpin"] = tiulpin_saliency_path
    except Exception as e:
        print(f"[Saliency-Tiulpin] WARNING: overlay failed: {e}")

    # Ours : Saliency maps for every patch in the bag
    try:
        with torch.no_grad():
            o_logits = ours_wrapped(torch.tensor(ours_x, device=DEVICE))
            o_y = o_logits.argmax(dim=1).cpu().numpy().astype(int)
        o_attr = generate_attributions(ours_wrapped, ours_x, o_y, DEVICE)
        o_saliency = o_attr["Saliency (SA)"]   # (N_patches, C, 16, 16)

        ours_saliency_path = "saliency_ours_on_dicom.png"
        visualize_saliency_ours_on_dicom(
            patient_id    = SAMPLE_ID,
            side          = SAMPLE_SIDE,
            saliency_maps = o_saliency,
            save_path     = ours_saliency_path,
        )
        saliency_paths["ours"] = ours_saliency_path
    except Exception as e:
        print(f"[Saliency-Ours] WARNING: overlay failed: {e}")

    return {

        "tiulpin": {"pred": t_pred, "true": t_true, "probs": t_probs},
        "ours":    {"pred": o_pred, "true": o_true, "probs": o_probs},
        "quantus": xai_results,
        "saliency_overlays": saliency_paths,
    }



if __name__ == "__main__":
    results = main()


