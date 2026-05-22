"""
FedUSD: Federated Dataset Distillation via Unified SVD-based Distribution Matching.

Main training script. Each federated client distills its local data into a compact
synthetic dataset; the server aggregates synthetic images across rounds and trains
a global model on the accumulated pool using linearly increasing sample weights.
"""

import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
import random
import time
from collections import defaultdict
from torchvision.utils import save_image
from tqdm import tqdm

from utils_train import *
from utils import *
from models import *
from dataset import *
from dataset.dataset_perlabel import *
from frequency_transforms import DCT


# ---------------------------------------------------------------------------
# Frequency-domain helper
# ---------------------------------------------------------------------------

def get_freq_image(img, mask, indices, resolution, device, need_copy=False):
    """Reconstruct spatial images from masked frequency coefficients via inverse DCT.

    Args:
        img (Tensor): Frequency-domain representation of synthetic images.
        mask (Tensor): Binary mask selecting active frequency components.
        indices (array-like): Indices of images to reconstruct.
        resolution (int): Spatial resolution (height == width).
        device (torch.device): Target device.
        need_copy (bool): If True, detach and deep-copy before reconstruction.

    Returns:
        Tensor: Reconstructed spatial images.
    """
    if need_copy:
        freq_selected = copy.deepcopy(img[indices].detach())
        mask_selected = copy.deepcopy(mask[indices].detach())
    else:
        freq_selected = img[indices]
        mask_selected = mask[indices]

    return DCT(resolution=resolution, device=device).inverse(mask_selected * freq_selected)


# ---------------------------------------------------------------------------
# Sliced Wasserstein Distance (pure PyTorch)
# ---------------------------------------------------------------------------

def compute_swd(logits1, logits2, num_projections=100):
    """Compute the Sliced Wasserstein Distance (SWD) between two 1-D tensors.

    Random unit projections are drawn; both tensors are projected, sorted, and
    the mean absolute difference of the sorted projections is returned as the SWD
    estimate (equivalent to averaging 1-D Wasserstein distances over directions).

    Args:
        logits1 (Tensor): 1-D source tensor.
        logits2 (Tensor): 1-D target tensor (moved to logits1's device internally).
        num_projections (int): Number of random projection directions.

    Returns:
        Tensor | float: Estimated SWD; 0.0 when input dimension is zero.
    """
    assert logits1.dim() == 1 and logits2.dim() == 1, "Inputs must be 1-D tensors"
    D = logits1.shape[0]
    if D == 0:
        return 0.0

    device  = logits1.device
    logits2 = logits2.to(device)

    # Sample unit random projections
    projections = torch.randn(num_projections, D, device=device)
    projections = projections / projections.norm(dim=1, keepdim=True)

    # Project, sort, and compute mean absolute difference
    sorted_proj1, _ = torch.sort(torch.matmul(projections, logits1))
    sorted_proj2, _ = torch.sort(torch.matmul(projections, logits2))
    return torch.mean(torch.abs(sorted_proj1 - sorted_proj2))


# ---------------------------------------------------------------------------
# SVD-based distribution matching loss
# ---------------------------------------------------------------------------

def compare_commonality_and_individuality_tensor(A_maps, B_maps, k=1):
    """Compute the SVD-based distribution distance between two feature sets.

    Each set is decomposed via truncated SVD into a rank-k "common" subspace.
    The loss is the 1-D EMD (Wasserstein-1) distance between the dominant
    singular directions of A and B, computed as the mean absolute difference
    of their sorted values (closed-form quantile-matching solution).

    Args:
        A_maps (Tensor): Shape (N1, C, H, W) or (N1, D) -- synthetic features.
        B_maps (Tensor): Shape (N2, C, H, W) or (N2, D) -- real features.
        k (int): Number of dominant singular vectors to retain.

    Returns:
        Tensor: Scalar EMD distance between the top singular directions of A and B.
    """

    def svd_decompose(X, k):
        """Truncated SVD; falls back to CPU low-rank SVD on numerical failure."""
        if not torch.isfinite(X).all():
            X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except RuntimeError:
            # GPU SVD can fail for ill-conditioned matrices; retry on CPU
            X_cpu = X.detach().to("cpu", dtype=torch.float32)
            U, S, Vh = torch.svd_lowrank(X_cpu, q=k + 5, niter=2)
            Vh = Vh.T

        U_k  = U[:, :k]     # (N, k)
        S_k  = S[:k]        # (k,)
        Vh_k = Vh[:k, :]    # (k, D)

        X_common     = (U_k @ torch.diag(S_k) @ Vh_k).cuda()
        X_individual = X - X_common
        energy_ratio = (S ** 2) / (S ** 2).sum()
        return X_common, X_individual, Vh_k, energy_ratio

    # Flatten spatial dims for 4-D feature-map inputs
    if A_maps.dim() == 4:
        N1, C, H, W = A_maps.shape
        N2          = B_maps.shape[0]
        A_flat      = A_maps.reshape(N1, -1)
        B_flat      = B_maps.reshape(N2, -1)
    else:
        A_flat = A_maps
        B_flat = B_maps

    A_common, _, Vh_A, _ = svd_decompose(A_flat, k)
    B_common, _, Vh_B, _ = svd_decompose(B_flat, k)

    # 1-D EMD between the top singular directions (dominant distribution axes).
    # Sorting both vectors and comparing element-wise is the closed-form
    # Wasserstein-1 solution for equal-weight empirical distributions.
    emd_dist = torch.mean(torch.abs(torch.sort(Vh_A[0])[0] - torch.sort(Vh_B[0])[0]))

    return emd_dist


# ---------------------------------------------------------------------------
# Main federated distillation procedure
# ---------------------------------------------------------------------------

def main(seed000):
    """Run one full federated dataset distillation experiment.

    Args:
        seed000 (int): Random seed used for reproducibility logging.
    """
    # ------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='FedUSD: Federated Dataset Distillation')

    # Dataset / model
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('--model', type=str, default='ConvNet', help='Backbone model architecture')
    parser.add_argument('--ipc', type=int, default=10, help='Synthetic images per class')
    parser.add_argument('--data_path', type=str, default='data', help='Root path to datasets')

    # Evaluation
    parser.add_argument('--eval_mode', type=str, default='M',
                        help='Evaluation mode: S=same model, M=multi-arch, W=width, D=depth, '
                             'A=activation, P=pooling, N=normalization')
    parser.add_argument('--num_exp', type=int, default=1, help='Number of independent experiments')
    parser.add_argument('--num_eval', type=int, default=20,
                        help='Number of randomly-initialised models used for evaluation')
    parser.add_argument('--epoch_eval_train', type=int, default=300,
                        help='Training epochs for evaluation model on the synthetic set')

    # Distillation optimisation
    parser.add_argument('--Iteration', type=int, default=1000,
                        help='Inner optimisation iterations per federated round')
    parser.add_argument('--lr_img', type=float, default=1.0,
                        help='Learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01,
                        help='Learning rate for updating network weights')
    parser.add_argument('--batch_real', type=int, default=256,
                        help='Batch size for sampling real training data')
    parser.add_argument('--batch_train', type=int, default=64,
                        help='Batch size for training the global model')

    # Synthetic data initialisation
    parser.add_argument('--init', type=str, default='real',
                        help='Initialisation strategy: "real" | "noise" | "pretrained"')
    parser.add_argument('--init_path', type=str, default='',
                        help='Path to pre-trained checkpoint for "pretrained" init')

    # Experiment bookkeeping
    parser.add_argument('--dis_metric', type=str, default='ours',
                        help='Distribution distance metric identifier')
    parser.add_argument('--gpu', type=str, default='auto',
                        help='GPU ID(s); "auto" picks the least-loaded device')
    parser.add_argument('--save', type=str, default='./results_AF_0.1svhn',
                        help='Output directory prefix')
    parser.add_argument('--group', type=str, default='FedDM',
                        help='Experiment group name; shared parent folder for all exps in group')
    parser.add_argument('--tag', type=str, default='1',
                        help='Additional tag appended to the experiment ID')

    # Federated learning settings
    parser.add_argument('--num_users', type=int, default=10,
                        help='Total number of federated clients')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Fraction of clients participating in each round')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Dirichlet concentration parameter for non-IID data partitioning')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of federated communication rounds')
    parser.add_argument('--extreme', type=int, default=0, help='Enable extreme / fast mode')

    # Data augmentation flags
    parser.add_argument('--no_aug', type=int, default=0,
                        help='Disable augmentation during distillation and evaluation (1=off)')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Fast prototype mode (fewer evaluations)')
    parser.add_argument('--aug', type=bool, default=True,
                        help='Apply quadrant down-sampling to inflate the synthetic set size')

    # Gradient-matching loop counts
    parser.add_argument('--inner_loop', type=int, default=-1,
                        help='Gradient-matching iterations (upper level); -1 = auto from ipc')
    parser.add_argument('--outer_loop', type=int, default=-1,
                        help='Synthetic-data training steps (lower level); -1 = auto from ipc')

    # Gradient normalisation
    parser.add_argument('--match_norm', type=int, default=0,
                        help='Train reference network with normalised SGD for gradient matching')

    # Matching mode
    parser.add_argument('--match_mode', type=str, default='whole',
                        help='Feature matching granularity: "whole" or "per-label"')

    # SAM / ASAM perturbation
    parser.add_argument('--rho', type=float, default=0.5,
                        help='SAM perturbation radius (0.5 for SAM, 0.05 for ASAM)')
    parser.add_argument('--progress_perturb', type=int, default=0,
                        help='Gradually increase rho over training')
    parser.add_argument('--opt_X', default='sgd', type=str,
                        choices=['sam', 'asam', 'sgd', 'sam-rand', 'asam-rand'],
                        help='Optimiser for synthetic images (X)')
    parser.add_argument('--opt_net', default='sgd', type=str,
                        choices=['sam', 'asam', 'sgd', 'sam-rand', 'asam-rand'],
                        help='Optimiser for network weights (lower level and evaluation)')
    parser.add_argument('--opt_perturb', default='none', type=str,
                        choices=['none', 'sam', 'asam', 'sam-rand', 'asam-rand'],
                        help='Sharpness-aware perturbation applied during gradient matching')
    parser.add_argument('--weight_decay_net', default=0, type=float,
                        help='Weight decay for the network optimiser')

    # Differentiable Siamese Augmentation (DSA)
    parser.add_argument('--method', type=str, default='DSA', choices=['DC', 'DSA'],
                        help='Distillation method: DC (Dataset Condensation) or DSA')
    parser.add_argument('--dsa_strategy', type=str,
                        default='color_crop_cutout_flip_scale_rotate',
                        help='DSA augmentation pipeline specification')
    parser.add_argument('--opt_net_mom', type=float, default=0.5,
                        help='SGD momentum for network optimiser (0 in DSA, 0.5 in DC)')

    args = parser.parse_args()

    # Placeholder -- input normalisation handled downstream
    args.normalize_input = 'none'

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = (
        str(pick_gpu_lowest_memory()) if args.gpu == 'auto' else args.gpu
    )

    # Resolve loop counts from ipc when not explicitly specified
    outer_loop, inner_loop = get_loops(args.ipc)
    if args.outer_loop == -1:
        args.outer_loop = outer_loop
    if args.inner_loop == -1:
        args.inner_loop = inner_loop

    args.device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa       = args.method == 'DSA'
    args.aa        = False  # placeholder for AutoAugment (unused)

    if 'debug' in args.tag:
        args.group = 'debug'
    if 'dev' in args.tag:
        args.group = 'dev'

    # ------------------------------------------------------------------
    # Build experiment output path
    # ------------------------------------------------------------------
    exp_id  = args.save
    exp_id += f'_[{args.model}]'
    exp_id += f'_[{args.dataset}]'
    exp_id += f'_[ipc-{args.ipc}]'
    exp_id += f'_[loop={args.outer_loop}x{args.inner_loop}]'
    exp_id += f'_[{args.match_mode}]'
    exp_id += f'_[{args.method}]'
    rho_tag = f'{args.rho}up' if args.progress_perturb else f'{args.rho}'
    if args.init != 'noise':
        exp_id += f'_[init-{args.init}]'
    if args.opt_X != 'sgd':
        exp_id += f'_[X-{args.opt_X}]-{rho_tag}'
    if args.opt_net != 'sgd':
        exp_id += f'_[net-{args.opt_net}]'
    if args.opt_perturb != 'none':
        exp_id += f'_[gm-{args.opt_perturb}-{rho_tag}]'
    if args.batch_real != 256:
        exp_id += f'_[bsr={args.batch_real}]'
    if args.no_aug:
        exp_id += '[no_aug]'
    if args.method == 'DSA' and args.opt_net_mom != 0:
        exp_id += f'_[mom{args.opt_net_mom}]'
    if args.method == 'DC' and args.opt_net_mom != 0.5:
        exp_id += f'_[mom{args.opt_net_mom}]'
    if args.tag and args.tag != 'none':
        exp_id += f'_[tag-{args.tag}]'
    if args.tag == 'none':
        exp_id += f'_[num_users-{args.num_users}]_[frac-{args.frac}]_[extre-{args.extreme}]'
    if 'debug' in args.tag:
        exp_id = args.tag

    args.save = (
        os.path.join('experiments/', exp_id)
        if args.group == 'none'
        else os.path.join('experiments/', f'{args.group}/', exp_id)
    )

    args.ckpt_path = os.path.join(args.save, 'ckpts')
    args.vis_path  = os.path.join(args.save, 'vis')
    os.makedirs(args.ckpt_path, exist_ok=True)
    os.makedirs(args.vis_path,  exist_ok=True)

    # ------------------------------------------------------------------
    # Logging -- write to both stdout and a file
    # ------------------------------------------------------------------
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'), mode='w')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, \
        trainloader_full, testloader = get_dataset(args.dataset, args.data_path)
    del dst_test  # only the test loader is needed hereafter
    args.num_classes = num_classes

    # ------------------------------------------------------------------
    # Initialise per-client states (dataset wrappers + image buffers)
    # ------------------------------------------------------------------
    user_states  = {}
    exp_acc_list = []

    for exp in range(args.num_exp):
        # Partition training data across clients with a Dirichlet non-IID split
        dict_users, dict_classes = partition(dst_train, args.num_users, args.num_classes, args.alpha)
        logging.info(dict_classes)

        criterion = cross_entropy_loss_cust(args).cuda()  # soft-label cross-entropy

        # Synthetic label tensor (fixed hard labels, not learnable)
        label_syn = torch.tensor(
            [np.ones(args.ipc) * i for i in range(num_classes)],
            dtype=torch.long, requires_grad=False, device=args.device
        ).view(-1)

        for idx in range(args.num_users):
            data_idxs = dict_users[idx]
            classes   = dict_classes[idx]
            sub_train = DatasetSplit(dst_train, data_idxs)

            if args.dataset == 'ImageNet':
                dst_perlabel      = PerLabelLargeDataset(sub_train, num_classes, channel, args)
                imnet_loader      = dst_perlabel.loader
                imnet_iterator    = iter(imnet_loader)
                imnet_init_images = dst_perlabel.get_init_images(ipc=1)
            else:
                dst_perlabel = PerLabelDatasetNonIID(sub_train, classes, channel, args)

            # Placeholder synthetic buffer; will be overwritten before training
            image_syn = torch.randn(
                size=(len(classes) * args.ipc, channel, im_size[0], im_size[1]),
                dtype=torch.float, requires_grad=True, device=args.device
            )
            user_states[idx] = LocalUser(dst_perlabel, image_syn)

        logging.info('%s training begins' % get_time())

        fed_accs     = []
        global_model = get_network(args.model, channel, num_classes, im_size).cuda()
        all_img_syn  = []
        all_lbl_syn  = []
        all_weight   = []
        del dst_train

        # ==============================================================
        # Federated communication rounds
        # ==============================================================
        for curr_epoch in tqdm(range(args.epochs)):
            global_model.train()

            # Perturb a copy of the global model to act as the teacher network.
            # In round 0 a freshly-initialised random network is used instead.
            if curr_epoch != 0:
                net = random_perturb(copy.deepcopy(global_model))
            else:
                net = get_network(args.model, channel, num_classes, im_size).cuda()

            # ------------------------------------------------------------------
            # Build per-class mean log-softmax statistics on the full training
            # set. These serve as a reference distribution for distillation.
            # ------------------------------------------------------------------
            category_out0  = defaultdict(list)
            category_logit = defaultdict(list)

            for images, labels in trainloader_full:
                images, labels = images.cuda(), labels.cuda()
                _, _, logit    = net.embed(images)
                logit_softmax  = F.log_softmax(logit / 2, dim=1)
                for i in range(len(labels)):
                    category_out0[labels[i].item()].append(logit[i].detach().cpu())
                    category_logit[labels[i].item()].append(logit_softmax[i].detach().cpu())

            mean_out0  = {cls: torch.mean(torch.stack(v), dim=0) for cls, v in category_out0.items()}
            mean_logit = {cls: torch.mean(torch.stack(v), dim=0) for cls, v in category_logit.items()}

            # Per-class diagonal logit values used for curriculum weighting
            sorted_keys       = sorted(mean_logit.keys())
            mean_logit_tensor = torch.stack([mean_logit[k] for k in sorted_keys])
            logits_vector     = torch.tensor(
                [mean_logit_tensor[cls][cls].item() for cls in range(mean_logit_tensor.shape[0])]
            )

            args.dsa     = args.method == 'DSA'
            curr_img_syn = []
            curr_lbl_syn = []

            logging.info('\n================== Epoch %d ==================' % curr_epoch)
            m          = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            logging.info('\nChoosing users {}'.format(' '.join(map(str, idxs_users))))

            # ==============================================================
            # Per-client synthetic data distillation
            # ==============================================================
            for idx in idxs_users:
                user       = user_states[idx]
                classes    = dict_classes[idx]
                get_images = user.dataset.get_images

                # Re-initialise synthetic images at the start of each round
                image_syn = torch.randn(
                    size=(len(classes) * args.ipc, channel, im_size[0], im_size[1]),
                    dtype=torch.float, requires_grad=True, device=args.device
                )

                # --------------------------------------------------------------
                # Synthetic image initialisation strategies
                # --------------------------------------------------------------
                if args.init == 'real':
                    logging.info('Initialise synthetic data from real images')
                    if args.dataset == 'ImageNet':
                        image_syn.data.copy_(imnet_init_images.data)
                    else:
                        for i, c in enumerate(classes):
                            if not args.aug:
                                # Direct copy from real images
                                image_syn.data[i * args.ipc:(i + 1) * args.ipc] = \
                                    get_images(c, args.ipc).detach().data
                            else:
                                # Place down-sampled real images in each of the four quadrants
                                half = im_size[0] // 2
                                for rs, re, cs_, ce in [
                                    (0,    half,        0,    half),
                                    (half, im_size[0],  0,    half),
                                    (0,    half,        half, im_size[1]),
                                    (half, im_size[0],  half, im_size[1]),
                                ]:
                                    image_syn.data[i * args.ipc:(i + 1) * args.ipc, :, rs:re, cs_:ce] = \
                                        downscale(get_images(c, args.ipc), 0.5).detach().data

                elif args.init == 'pretrained':
                    logging.info('Initialise synthetic data from pre-trained checkpoint')
                    ckpt_path_user = os.path.join(args.ckpt_path, f'exp_{exp}', f'user_{idx}')
                    os.makedirs(ckpt_path_user, exist_ok=True)
                    prev_ckpt  = os.path.join(
                        ckpt_path_user,
                        f'run_{args.dataset}_{args.model}_{curr_epoch - 1}.pt'
                    )
                    syn_state = torch.load(prev_ckpt)
                    assert syn_state['data'][0].shape[0] == args.ipc * len(classes)
                    image_syn.data.copy_(syn_state['data'][0].to(args.device))
                    label_syn.data.copy_(syn_state['data'][1])

                else:  # 'noise' initialisation
                    logging.info('Initialise synthetic data from noise for user %d' % idx)
                    img_real = user.dataset.get_random_images(args.batch_real).detach().data
                    image_syn.requires_grad_(False)
                    # Scale noise amplitude to match real image statistics
                    image_syn[:, 0] = (
                        image_syn[:, 0] / image_syn[:, 0].abs().max()
                        * img_real[:, 0].abs().max()
                    )

                # Full-resolution frequency mask (all coefficients activated)
                mask = torch.zeros(
                    (len(classes) * args.ipc, channel, im_size[0], im_size[1]),
                    dtype=torch.float, device=args.device
                )
                mask[:, :, :32, :32] = 1.0
                mask.requires_grad   = False

                optimizer_img = get_optimizer(
                    [image_syn], args.opt_X,
                    lr=args.lr_img, weight_decay=0, rho=0, momentum=0.5
                )
                optimizer_img.zero_grad()

                # --------------------------------------------------------------
                # Inner distillation loop
                # --------------------------------------------------------------
                for it in range(args.Iteration):
                    loss_avg = 0

                    # Refresh the teacher network each iteration
                    if curr_epoch != 0:
                        net = random_perturb(copy.deepcopy(global_model))
                    else:
                        net = get_network(args.model, channel, num_classes, im_size).cuda()

                    net.train()
                    for param in net.parameters():
                        param.requires_grad = False
                    embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed

                    # BN_flag: when True the network contains BatchNorm layers and real
                    # / synthetic images must be forwarded together so that BN statistics
                    # are computed over a mixed batch.
                    BN_flag = False

                    loss     = torch.tensor(0.0).cuda()
                    labs_syn = torch.LongTensor([]).cuda()

                    if not BN_flag:
                        # --------------------------------------------------
                        # Per-class feature matching (standard, no BatchNorm)
                        # --------------------------------------------------
                        criterion = nn.CrossEntropyLoss().to(args.device)

                        for i, c in enumerate(classes):
                            img_real     = get_images(c, args.batch_real)
                            img_syn      = image_syn[i * args.ipc:(i + 1) * args.ipc].reshape(
                                args.ipc, channel, im_size[0], im_size[1]
                            )
                            lab_syn      = torch.ones((args.ipc,),        device=args.device, dtype=torch.long) * c
                            lab_syn_real = torch.ones((args.batch_real,), device=args.device, dtype=torch.long) * c

                            if args.aug:
                                img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)
                            if args.dsa:
                                # Use a time-based seed so every iteration gets a fresh
                                # but reproducible augmentation pair for real and syn.
                                seed     = int(time.time() * 1000) % 100000
                                img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                                img_syn  = DiffAugment(img_syn,  args.dsa_strategy, seed=seed, param=args.dsa_param)

                            # Extract intermediate features via the (frozen) teacher network
                            _, output_real, logit_real = embed(img_real)
                            _, output_syn,  logit_syn  = embed(img_syn)

                            # SVD-based distribution distance between synthetic and real features
                            distance = compare_commonality_and_individuality_tensor(
                                output_syn, output_real, k=1
                            )

                            labs_syn = torch.cat([labs_syn, lab_syn], dim=0)

                            # Loss term 1: SVD-based dominant-subspace alignment
                            loss += 0.01 * distance
                            # Loss term 2: per-dimension variance alignment via 1-D EMD
                            var_syn_sorted  = torch.sort(torch.var(output_syn,  dim=0))[0]
                            var_real_sorted = torch.sort(torch.var(output_real, dim=0))[0]
                            loss += 0.01 * torch.mean(torch.abs(var_syn_sorted - var_real_sorted))

                    else:
                        # --------------------------------------------------
                        # BatchNorm-aware branch: forward all classes together
                        # so BN running statistics reflect the full batch.
                        # --------------------------------------------------
                        images_real_all = []
                        images_syn_all  = []

                        for i, c in enumerate(classes):
                            img_real = get_images(c, args.batch_real)
                            img_syn  = image_syn[i * args.ipc:(i + 1) * args.ipc].reshape(
                                args.ipc, channel, im_size[0], im_size[1]
                            )
                            if args.aug:
                                img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)
                            if args.dsa:
                                seed     = int(time.time() * 1000) % 100000
                                img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                                img_syn  = DiffAugment(img_syn,  args.dsa_strategy, seed=seed, param=args.dsa_param)
                            images_real_all.append(img_real)
                            images_syn_all.append(img_syn)

                        images_real_all = torch.cat(images_real_all, dim=0)
                        images_syn_all  = torch.cat(images_syn_all,  dim=0)

                        output_real = embed(images_real_all).detach()
                        output_syn  = embed(images_syn_all)
                        distance    = compare_commonality_and_individuality_tensor(
                            output_syn, output_real, k=1
                        )

                        # Loss term 1: SVD-based dominant-subspace alignment
                        loss += 0.01 * distance
                        # Loss term 2: per-dimension variance alignment via 1-D EMD
                        var_syn_sorted  = torch.sort(torch.var(output_syn,  dim=0))[0]
                        var_real_sorted = torch.sort(torch.var(output_real, dim=0))[0]
                        loss += 0.01 * torch.mean(torch.abs(var_syn_sorted - var_real_sorted))

                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()

                    loss_avg += loss.item()
                    loss_avg /= len(classes)
                    if (it + 1) % 500 == 0 or it == args.Iteration:
                        logging.info('%s user %d loss = %.4f at iteration %d' % (
                            get_time(), idx, loss_avg, it
                        ))

                # --------------------------------------------------------------
                # Collect per-client distilled images for this round
                # --------------------------------------------------------------
                image_syn_all = None
                label_syn_all = None
                for i, c in enumerate(classes):
                    img_syn = image_syn[i * args.ipc:(i + 1) * args.ipc].reshape(
                        args.ipc, channel, im_size[0], im_size[1]
                    )
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                    if args.aug:
                        img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)
                    image_syn_all = img_syn if i == 0 else torch.cat((image_syn_all, img_syn), dim=0)
                    label_syn_all = lab_syn if i == 0 else torch.cat((label_syn_all, lab_syn), dim=0)

                curr_img_syn.append(copy.deepcopy(image_syn_all.detach()))
                curr_lbl_syn.append(copy.deepcopy(label_syn_all.detach()))

                # --------------------------------------------------------------
                # Visualise distilled images and save checkpoint
                # --------------------------------------------------------------
                exp_user_path = os.path.join(args.vis_path, f'exp_{exp}', f'user_{idx}')
                os.makedirs(exp_user_path, exist_ok=True)
                save_name = os.path.join(
                    exp_user_path,
                    f'vis_{args.dataset}_{args.model}_{args.ipc}ipc_epoch_{curr_epoch}.png'
                )
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis = image_syn_vis.clamp(0.0, 1.0)
                save_image(image_syn_vis, save_name, nrow=args.ipc)

                ckpt_user_path = os.path.join(args.ckpt_path, f'exp_{exp}', f'user_{idx}')
                os.makedirs(ckpt_user_path, exist_ok=True)
                torch.save(
                    {'data': [
                        copy.deepcopy(image_syn_all.detach().cpu()),
                        copy.deepcopy(labs_syn.detach().cpu()),
                    ]},
                    os.path.join(ckpt_user_path, f'run_{args.dataset}_{args.model}_{curr_epoch}.pt')
                )

            # ------------------------------------------------------------------
            # Aggregate the synthetic pool and update the global model
            # ------------------------------------------------------------------
            all_img_syn.extend(curr_img_syn)
            all_lbl_syn.extend(curr_lbl_syn)

            # Sliding-window retention: keep only the most recent 3 rounds to
            # prevent the pool from growing unboundedly.
            if args.aug and curr_epoch > 2:
                keep_start  = int(1 * args.num_users * args.frac)
                keep_end    = int(4 * args.num_users * args.frac)
                all_img_syn = all_img_syn[keep_start:keep_end]
                all_lbl_syn = all_lbl_syn[keep_start:keep_end]

            # Resolve augmentation parameters for global model training
            if args.dsa:
                args.epoch_eval_train = 500
                args.dc_aug_param     = None
                logging.info('DSA augmentation strategy: \n%s' % args.dsa_strategy)
                logging.info('DSA augmentation parameters: \n%s' % args.dsa_param.__dict__)
            else:
                args.dc_aug_param = get_daparam(args.dataset, args.model, 'ConvNet')
                logging.info('DC augmentation parameters: \n%s' % args.dc_aug_param)

            args.epoch_eval_train = 500  # Use 500 epochs whenever augmentation is active

            global_model.train()
            start_time = time.time()

            all_img_syn_eval = torch.cat(all_img_syn, dim=0)
            all_lbl_syn_eval = torch.cat(all_lbl_syn, dim=0)
            logging.info('Aggregated synthetic set shape: %s' % str(all_img_syn_eval.shape))

            # Linearly increasing sample weights reward more recently distilled images
            if curr_epoch == 0:
                num_img_per_round = all_img_syn_eval.shape[0]
            weights         = torch.ones(num_img_per_round).cuda() * (curr_epoch + 1)
            all_weight.append(weights)
            all_weight_eval = torch.cat(all_weight)

            global_model, acc_syns_train, acc_full_test = evaluate_synset(
                curr_epoch, global_model,
                all_img_syn_eval, all_lbl_syn_eval,
                testloader, args, logits_vector,
                weight=all_weight_eval
            )
            logging.info('%s Epoch = %04d test acc = %.4f' % (get_time(), curr_epoch, acc_full_test))
            logging.info('Global model training time: %.6f s' % (time.time() - start_time))
            fed_accs.append(acc_full_test)

        exp_acc_list.append(fed_accs)

    # ------------------------------------------------------------------
    # Save and report final results
    # ------------------------------------------------------------------
    exp_acc_list = np.array(exp_acc_list)
    acc_mean     = np.mean(exp_acc_list, axis=0)
    acc_std      = np.std(exp_acc_list,  axis=0)

    results_path = os.path.join(os.getcwd(), 'results_AF_0.1svhn.txt')
    with open(results_path, 'a') as f:
        f.write(f'seed000: {seed000}, acc_mean: {acc_mean}\n')

    logging.info(acc_mean)
    logging.info(acc_std)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    seed000 = 1  # Fixed seed for reproducibility

    random.seed(seed000)
    np.random.seed(seed000)
    torch.manual_seed(seed000)
    torch.cuda.manual_seed(seed000)
    torch.cuda.manual_seed_all(seed000)
    torch.backends.cudnn.deterministic = True  # Force deterministic CUDA kernels
    torch.backends.cudnn.benchmark     = False  # Disable cuDNN auto-tuner

    main(seed000=seed000)
