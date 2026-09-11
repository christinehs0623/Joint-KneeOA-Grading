"""
Dataset tools


(c) Aleksei Tiulpin, University of Oulu, 2017
"""

import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import KFold


def get_pair(I):
    """
    Generates pair of images 128x128 from the knee joint.
    ps shows how big area should be mapped into that region.
    """
    s = I.size[0]
    pad = int(np.floor(s/3))
    ps = 128

    l = I.crop([0, pad, ps, pad+ps])
    m = I.crop([s-ps, pad, s, pad+ps])
    m = m.transpose(Image.FLIP_LEFT_RIGHT)
    
    return l, m


def make_splits(data_root, seed=42, fold=0, n_folds=5, test_frac=0.2):
    """
    Reads a folder structured as:
        data_root/
            0/  <- KL grade 0 images
            1/
            2/
            3/
            4/

    Returns three lists of (absolute_filepath, label) tuples:
        train_items, val_items, test_items

    Splitting strategy (applied independently per KL class):
        1. A fixed 20% held-out test set is selected using `seed`.
           This set never changes regardless of `fold`.
        2. The remaining 80% is split into train/val using
           `n_folds`-fold cross-validation.  The `fold` argument
           (0 … n_folds-1) selects which fold to use as validation.

    Every pool sample appears exactly once in validation across all folds.
    """
    # Dedicated RNG for the fixed test split (depends only on seed)
    rng_test = np.random.RandomState(seed)

    train_items, val_items, test_items = [], [], []

    for kl in range(5):
        class_dir = os.path.join(data_root, str(kl))
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}"
            )
        files = sorted(os.listdir(class_dir))
        files = [f for f in files if not f.startswith('.')]  # skip hidden files
        files = np.array(files)

        # --- Stage 1: fixed test set (determined only by seed) ---
        idx = np.arange(len(files))
        rng_test.shuffle(idx)
        n_test = int(np.floor(len(files) * test_frac))
        test_idx = idx[:n_test]
        pool_idx = idx[n_test:]

        test_f = files[test_idx]
        pool_f = files[pool_idx]

        # --- Stage 2: k-fold CV on the pool ---
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(pool_f))
        train_rel, val_rel = splits[fold]

        train_f = pool_f[train_rel]
        val_f   = pool_f[val_rel]

        for fname in train_f:
            train_items.append((os.path.join(class_dir, fname), kl))
        for fname in val_f:
            val_items.append((os.path.join(class_dir, fname), kl))
        for fname in test_f:
            test_items.append((os.path.join(class_dir, fname), kl))

    return train_items, val_items, test_items


class KneeGradingDataset(data.Dataset):
    """
    Dataset class.

    Parameters
    ----------
    items : list of (filepath, label) tuples
        Each element is an absolute path to an image file and its KL-grade label.
    transform : callable
        Transform applied to each 128x128 patch (e.g. ToTensor + Normalize).
    augment : callable or None
        Optional augmentation applied to the full image before cropping.
    """
    def __init__(self, items, transform, augment=None):
        self.items = items
        self.transform = transform
        self.augment = augment

    def __getitem__(self, index):
        fpath, target = self.items[index]

        img = Image.open(fpath).convert("L")   # optional but ensures grayscale

        if self.augment is not None:
            img = self.augment(img)

        l, m = get_pair(img)

        l = self.transform(l)
        m = self.transform(m)

        return l, m, target, fpath

    def __len__(self):
        return len(self.items)


class LimitedRandomSampler(data.sampler.Sampler):
    """
    Allows to use limited number of batches in the training
    """
    def __init__(self, data_source, nb, bs):
        self.data_source = data_source
        self.n_batches = nb
        self.bs = bs

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long()[:self.n_batches*self.bs])

    def __len__(self):
        return self.n_batches*self.bs
