"""
Main training script

(c) Aleksei Tiulpin, University of Oulu, 2017

"""

from __future__ import print_function

import argparse
import os
import gc
import pickle
import time

from termcolor import colored

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from visdom import Visdom

cudnn.benchmark = True

from ouludeepknee.train.dataset import KneeGradingDataset, LimitedRandomSampler, make_splits
from ouludeepknee.train.train_utils import train_epoch, adjust_learning_rate
from ouludeepknee.train.val_utils import validate_epoch
from ouludeepknee.train.model import KneeNet
from ouludeepknee.train.augmentation import (CenterCrop, CorrectGamma, Jitter, Rotate, CorrectBrightness, CorrectContrast)


SNAPSHOTS_KNEE_GRADING = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../snapshots_knee_grading'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/data/net/datasets/OAI_Extracted/images/xray/preprocess_Knee_Radiographs_Tiulpin_png',
                        help='Path to folder containing class subfolders 0-4')
    parser.add_argument('--snapshots',  default=SNAPSHOTS_KNEE_GRADING)
    parser.add_argument('--experiment',  default='own_net')
    parser.add_argument('--patch_size', type=int, default=130)
    parser.add_argument('--base_width', type=int, default=32)
    parser.add_argument('--start_val', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_drop', type=int, default=20)
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=5e-5)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--val_bs', type=int, default=8)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--bootstrap', type=int, default=1)
    parser.add_argument('--n_batches', type=int, default=-1)
    parser.add_argument('--n_threads', type=int, default=20)
    parser.add_argument('--use_visdom', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0,
                        help='CV fold index (0 to n_folds-1)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    args = parser.parse_args()
    cur_lr = args.lr

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.isdir(args.snapshots):
        os.mkdir(args.snapshots)

    cur_snapshot = time.strftime('%Y_%m_%d_%H_%M_%S')
    os.mkdir(os.path.join(args.snapshots, cur_snapshot))
    with open(os.path.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # -----------------------------------------------------------------------
    # Build fixed test set + stratified k-fold CV splits from class subfolders 0-4
    # -----------------------------------------------------------------------
    print(colored('==> ', 'green') +
          f'Building splits (fold {args.fold}/{args.n_folds}) from: ' + args.data)
    train_items, val_items, test_items = make_splits(
        args.data, seed=args.seed, fold=args.fold,
        n_folds=args.n_folds, test_frac=0.2
    )

    # Print split sizes per class for transparency
    for kl in range(5):
        n_tr = sum(1 for _, lbl in train_items if lbl == kl)
        n_va = sum(1 for _, lbl in val_items   if lbl == kl)
        n_te = sum(1 for _, lbl in test_items  if lbl == kl)
        print(colored('==> ', 'green') +
              f'  KL {kl}: train={n_tr}, val={n_va}, test={n_te}')

    # -----------------------------------------------------------------------
    # Oversampling: balance classes in the train set
    # -----------------------------------------------------------------------
    # Group train items by class label for per-epoch oversampling
    train_by_class = {kl: [] for kl in range(5)}
    for item in train_items:
        train_by_class[item[1]].append(item)

    train_cats_length = [len(train_by_class[kl]) for kl in range(5)]
    oversample_size = int(sum(train_cats_length) / 5)
    print(colored('==> ', 'green') + f'Oversample target size per class: {oversample_size}')

    # -----------------------------------------------------------------------
    # Estimate channel mean / std on the (oversampled) training set
    # -----------------------------------------------------------------------
    if os.path.isfile(os.path.join(args.snapshots, 'mean_std.npy')):
        tmp = np.load(os.path.join(args.snapshots, 'mean_std.npy'))
        mean_vector, std_vector = tmp
    else:
        transf_tens = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.float()
        ])

        # Build an initial oversampled train list for mean/std estimation
        np.random.seed(args.seed)
        init_train_items = []
        for kl in range(5):
            items_kl = np.array(train_by_class[kl], dtype=object)
            chosen_idx = np.random.choice(len(items_kl), size=oversample_size, replace=True)
            init_train_items.extend(items_kl[chosen_idx].tolist())

        train_ds_ms = KneeGradingDataset(init_train_items, transform=transf_tens)
        train_loader_ms = data.DataLoader(train_ds_ms, batch_size=args.bs,
                                          num_workers=args.n_threads)

        mean_vector = np.zeros(1)
        std_vector  = np.zeros(1)

        print(colored('==> ', 'green') + 'Estimating the mean')
        pbar = tqdm(total=len(train_loader_ms),  disable=True)
        for entry in train_loader_ms:
            batch_l = entry[0]
            batch_m = entry[1]
            for j in range(mean_vector.shape[0]):
                mean_vector[j] += (batch_l[:, j, :, :].mean() + batch_m[:, j, :, :].mean()) / 2.
                std_vector[j]  += (batch_l[:, j, :, :].std()  + batch_m[:, j, :, :].std())  / 2.
            pbar.update()
        mean_vector /= len(train_loader_ms)
        std_vector  /= len(train_loader_ms)
        np.save(os.path.join(args.snapshots, 'mean_std.npy'), [mean_vector, std_vector])
        pbar.close()

    print(colored('==> ', 'green') + 'Mean: ', mean_vector)
    print(colored('==> ', 'green') + 'Std: ',  std_vector)

    # -----------------------------------------------------------------------
    # Transforms
    # -----------------------------------------------------------------------
    normTransform = transforms.Normalize(mean_vector, std_vector)
    patch_transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.float(),
        normTransform,
    ])
    augment_transforms = transforms.Compose([
        CorrectBrightness(0.7,1.3),
        CorrectContrast(0.7,1.3),
        CorrectGamma(0.5,2.5,res=8),     
    ])
    # -----------------------------------------------------------------------
    # Validation loader (fixed; no oversampling)
    # -----------------------------------------------------------------------
    val_ds = KneeGradingDataset(val_items, transform=patch_transform)
    val_loader = data.DataLoader(val_ds,
                                 batch_size=args.val_bs,
                                 num_workers=args.n_threads)

    print(colored('==> ', 'blue') + 'Initialized the loaders....')

    # -----------------------------------------------------------------------
    # Network, optimizer, criterion
    # -----------------------------------------------------------------------
    # net = nn.DataParallel(KneeNet(args.base_width, args.drop, True))
    # net.cuda()
    net = KneeNet(args.base_width, args.drop, True).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = F.cross_entropy

    # Visdom
    if args.use_visdom:
        vis = Visdom()
    else:
        vis = None
    win = None
    win_metrics = None

    train_losses = []
    val_losses   = []
    val_mse      = []
    val_kappa    = []
    val_acc      = []

    best_kappa = -1
    prev_model = None

    train_started = time.time()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(args.n_epoch):

        # Per-epoch oversampling: re-sample from each class folder
        np.random.seed(args.seed + epoch)
        epoch_train_items = []
        for kl in range(5):
            items_kl = np.array(train_by_class[kl], dtype=object)
            chosen_idx = np.random.choice(
                len(items_kl),
                size=oversample_size * args.bootstrap,
                replace=True
            )
            epoch_train_items.extend(items_kl[chosen_idx].tolist())

        # Shuffle the combined oversampled list
        np.random.shuffle(epoch_train_items)

        train_ds = KneeGradingDataset(epoch_train_items, transform=patch_transform,
                                       augment=augment_transforms)

        N_batches = None
        if args.n_batches > 0:
            N_batches = args.n_batches

        if N_batches is not None:
            train_loader = data.DataLoader(
                train_ds, batch_size=args.bs,
                num_workers=args.n_threads,
                sampler=LimitedRandomSampler(train_ds, N_batches, args.bs)
            )
        else:
            train_loader = data.DataLoader(
                train_ds,
                batch_size=args.bs,
                num_workers=args.n_threads,
                shuffle=True
            )

        print(colored('==> ', 'blue') + 'Epoch:', epoch + 1, cur_snapshot)
        optimizer, cur_lr = adjust_learning_rate(optimizer, epoch + 1, args)
        print(colored('==> ', 'red') + 'LR:', cur_lr)

        start = time.time()
        train_loss = train_epoch(epoch, net, optimizer, train_loader, criterion, args.n_epoch)
        epoch_time = np.round(time.time() - start, 4)
        print(colored('==> ', 'green') + 'Epoch training time: {} s.'.format(epoch_time))

        if epoch >= args.start_val:
            start = time.time()
            val_loss, probs, truth, _ = validate_epoch(net, val_loader, criterion)

            preds = probs.argmax(1)
            cm    = confusion_matrix(truth, preds)
            kappa = np.round(cohen_kappa_score(truth, preds, weights="quadratic"), 4)
            acc   = np.round(np.mean(cm.diagonal().astype(float) / cm.sum(axis=1)), 4)
            mse   = np.round(mean_squared_error(truth, preds), 4)
            val_time = np.round(time.time() - start, 4)

            print(colored('==> ', 'green') + 'Kappa:', kappa)
            print(colored('==> ', 'green') + 'Avg. class accuracy', acc)
            print(colored('==> ', 'green') + 'MSE', mse)
            print(colored('==> ', 'green') + 'Val loss:', val_loss)
            print(colored('==> ', 'green') + 'Epoch val time: {} s.'.format(val_time))

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_mse.append(mse)
            val_acc.append(acc)
            val_kappa.append(kappa)

        # Visdom visualisation
        if epoch > args.start_val + 1 and args.use_visdom:
            if win is None:
                win = vis.line(
                    X=np.column_stack((np.arange(epoch, epoch + 2), np.arange(epoch, epoch + 2))),
                    Y=np.column_stack((np.array(train_losses[-2:]), np.array(val_losses[-2:]))),
                    opts=dict(
                        title='[{}]\nTrain / val loss [{}]'.format(args.experiment, cur_snapshot),
                        legend=['Train', 'Validation'])
                )
            else:
                vis.line(
                    X=np.column_stack((np.arange(epoch, epoch + 2), np.arange(epoch, epoch + 2))),
                    Y=np.column_stack((np.array(train_losses[-2:]), np.array(val_losses[-2:]))),
                    win=win,
                    update='append'
                )

            if win_metrics is None:
                win_metrics = vis.line(
                    X=np.column_stack((np.arange(epoch, epoch + 2),) * 3),
                    Y=np.column_stack((1 - np.array(val_mse[-2:]),
                                       np.array(val_kappa[-2:]),
                                       np.array(val_acc[-2:]))),
                    opts=dict(
                        title='[{}]\nMetrics[{}]'.format(args.experiment, cur_snapshot),
                        legend=['1-MSE', 'Kappa', 'Accuracy'])
                )
            else:
                vis.line(
                    X=np.column_stack((np.arange(epoch, epoch + 2),) * 3),
                    Y=np.column_stack((1 - np.array(val_mse[-2:]),
                                       np.array(val_kappa[-2:]),
                                       np.array(val_acc[-2:]))),
                    win=win_metrics,
                    update='append'
                )

        # Save logs
        np.save(os.path.join(args.snapshots, cur_snapshot, 'logs.npy'),
                [train_losses, val_losses, val_mse, val_acc, val_kappa])

        if epoch >= args.start_val:
            cur_snapshot_name = os.path.join(
                args.snapshots, cur_snapshot, 'epoch_{}.pth'.format(epoch + 1))
            if prev_model is None:
                torch.save(net.state_dict(), cur_snapshot_name)
                prev_model = cur_snapshot_name
                best_kappa = kappa
            else:
                if kappa > best_kappa:
                    os.remove(prev_model)
                    best_kappa = kappa
                    print('Saved snapshot:', cur_snapshot_name)
                    torch.save(net.state_dict(), cur_snapshot_name)
                    prev_model = cur_snapshot_name

        gc.collect()

    print(args.seed, 'Training took:', time.time() - train_started, 'seconds')

    # -----------------------------------------------------------------------
    # Final evaluation on the held-out test set
    # -----------------------------------------------------------------------
    print(colored('==> ', 'yellow') + 'Evaluating on the test set...')
    test_ds = KneeGradingDataset(test_items, transform=patch_transform)
    test_loader = data.DataLoader(test_ds,
                                  batch_size=args.val_bs,
                                  num_workers=args.n_threads)
    net.load_state_dict(torch.load(prev_model, map_location="cuda"))
    net.eval()

    with torch.no_grad():
        test_loss, test_probs, test_truth, _ = validate_epoch(
            net, test_loader, criterion
        )

    test_preds = test_probs.argmax(1)
    test_cm    = confusion_matrix(test_truth, test_preds)
    test_kappa = np.round(cohen_kappa_score(test_truth, test_preds, weights="quadratic"), 4)
    test_acc   = np.round(np.mean(test_cm.diagonal().astype(float) / test_cm.sum(axis=1)), 4)
    test_mse   = np.round(mean_squared_error(test_truth, test_preds), 4)
    test_f1 = np.round(f1_score(test_truth, test_preds, average="macro"), 4)

    plt.figure(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(test_truth, test_preds, normalize="true", cmap=plt.cm.Blues, values_format=".2f",
                                            display_labels=[0, 1, 2, 3, 4])

    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(args.snapshots, cur_snapshot, 'confusion_matrix.png'))
    plt.close()

    print(colored('==> ', 'yellow') + 'Test Kappa:', test_kappa)
    print(colored('==> ', 'yellow') + 'Test Avg. class accuracy:', test_acc)
    print(colored('==> ', 'yellow') + 'Test MSE:', test_mse)
    print(colored('==> ', 'yellow') + 'Test loss:', test_loss)
    print(colored('==> ', 'yellow') + 'Test f1:', test_f1)

    np.save(
        os.path.join(args.snapshots, cur_snapshot, "test_results.npy"),
        {
            "kappa": test_kappa,
            "balanced_accuracy": test_acc,
            "mse": test_mse,
            "loss": test_loss,
            "macro_f1": test_f1,
        },
        allow_pickle=True,
    )
