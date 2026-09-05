# -*- coding: utf-8 -*-
"""
MNIST 3/5 and 3/5/8 few-shot experiment:
QRF-Net vs parameter-matched CNN, CNN_full, and LeNet.

V7 stable-20 edition:
1) keep the original MNIST35/MNIST358 presets for reproducibility;
2) add mnist35_stable and mnist358_stable presets for 20-seed reruns;
3) use a safer gated logit-residual quantum branch with two data re-uploads;
4) apply longer linear warmup and EMA to reduce bad-seed collapses;
5) aggregate repeated random-seed runs for stable same-seed comparison.
"""

import argparse
import copy
import json
import math
import random
import sys
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset


# =========================================================
# 1. Utils
# =========================================================
def safe_logit(p: float, eps: float = 1e-6) -> float:
    """Convert a probability in (0, 1) to logit."""
    p = max(min(float(p), 1.0 - eps), eps)
    return math.log(p / (1.0 - p))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_mean(xs):
    xs = [float(x) for x in xs if not np.isnan(float(x))]
    return float(np.mean(xs)) if xs else float("nan")


def safe_std(xs):
    xs = [float(x) for x in xs if not np.isnan(float(x))]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def get_dataset_class(name: str):
    name = name.lower()
    if name == "mnist":
        return torchvision.datasets.MNIST
    if name == "fashionmnist":
        return torchvision.datasets.FashionMNIST
    if name == "cifar10":
        return torchvision.datasets.CIFAR10
    raise ValueError(f"Unsupported dataset: {name}")


def get_dataset_stats(name: str):
    name = name.lower()
    if name == "mnist":
        return (0.1307,), (0.3081,)
    if name == "fashionmnist":
        return (0.2860,), (0.3530,)
    if name == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    raise ValueError(f"Unsupported dataset: {name}")


def get_in_channels(name: str) -> int:
    return 3 if name.lower() == "cifar10" else 1


def build_transforms(dataset_name: str, aug_mode: str = "light"):
    mean, std = get_dataset_stats(dataset_name)
    is_cifar = dataset_name.lower() == "cifar10"

    if is_cifar:
        if aug_mode == "none":
            train_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif aug_mode == "light":
            train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif aug_mode == "medium":
            train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
            ])
        else:
            raise ValueError(f"Unsupported aug_mode: {aug_mode}")
    else:
        if aug_mode == "none":
            train_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif aug_mode == "light":
            train_tf = transforms.Compose([
                transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif aug_mode == "medium":
            train_tf = transforms.Compose([
                transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.96, 1.04)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            raise ValueError(f"Unsupported aug_mode: {aug_mode}")

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# =========================================================
# 2. Dataset
# =========================================================
class RemappedSubset(Dataset):
    def __init__(self, subset: Subset, class_ids: Tuple[int, ...]):
        self.subset = subset
        self.mapping = {c: i for i, c in enumerate(class_ids)}

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.mapping[int(y)]


def build_fewshot_subsets(
    dataset_name: str,
    data_root: str,
    class_ids: Tuple[int, ...],
    k_shot: int,
    val_shot: int,
    seed: int,
    aug_mode: str = "light",
    overfit_small: bool = False,
):
    train_tf, eval_tf = build_transforms(dataset_name, aug_mode=aug_mode)
    ds_class = get_dataset_class(dataset_name)

    train_full_train = ds_class(root=data_root, train=True, download=True, transform=train_tf)
    train_full_eval = ds_class(root=data_root, train=True, download=True, transform=eval_tf)
    test_full_eval = ds_class(root=data_root, train=False, download=True, transform=eval_tf)

    train_targets = torch.as_tensor(train_full_train.targets).cpu().numpy()
    test_targets = torch.as_tensor(test_full_eval.targets).cpu().numpy()

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for c in class_ids:
        idx = np.where(train_targets == c)[0]
        rng.shuffle(idx)
        if k_shot + val_shot > len(idx):
            raise ValueError(
                f"class {c} does not have enough samples for "
                f"k_shot={k_shot}, val_shot={val_shot}"
            )
        train_indices.extend(idx[:k_shot].tolist())
        val_indices.extend(idx[k_shot:k_shot + val_shot].tolist())

    if overfit_small:
        train_subset = Subset(train_full_train, train_indices)
        val_subset = Subset(train_full_eval, train_indices)
        test_subset = Subset(train_full_eval, train_indices)
    else:
        test_indices = []
        for c in class_ids:
            idx = np.where(test_targets == c)[0]
            test_indices.extend(idx.tolist())

        train_subset = Subset(train_full_train, train_indices)
        val_subset = Subset(train_full_eval, val_indices)
        test_subset = Subset(test_full_eval, test_indices)

    return train_subset, val_subset, test_subset


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(args, seed: int):
    train_subset, val_subset, test_subset = build_fewshot_subsets(
        dataset_name=args.dataset,
        data_root=args.data_root,
        class_ids=tuple(args.class_ids),
        k_shot=args.k_shot,
        val_shot=args.val_shot,
        seed=seed,
        aug_mode=args.aug_mode,
        overfit_small=args.overfit_small,
    )

    train_ds = RemappedSubset(train_subset, tuple(args.class_ids))
    val_ds = RemappedSubset(val_subset, tuple(args.class_ids))
    test_ds = RemappedSubset(test_subset, tuple(args.class_ids))

    pin = args.device.startswith("cuda") and torch.cuda.is_available()
    loader_seed = int(seed) + int(args.loader_seed_offset) + 1009 * int(args.k_shot)
    generator = torch.Generator()
    generator.manual_seed(loader_seed)
    num_workers = max(0, int(args.num_workers))
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, generator=generator,
        worker_init_fn=seed_worker, persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        worker_init_fn=seed_worker, persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        worker_init_fn=seed_worker, persistent_workers=persistent_workers
    )
    return train_loader, val_loader, test_loader


# =========================================================
# 3. Quantum part
# =========================================================
def quantum_readout_terms(n_qubits: int, readout_mode: str = "adjacent") -> List[Tuple[int, ...]]:
    singles = [(i,) for i in range(n_qubits)]
    if readout_mode == "z":
        return singles
    if readout_mode == "adjacent":
        return singles + [(i, i + 1) for i in range(n_qubits - 1)]
    if readout_mode == "all_pairs":
        return singles + list(combinations(range(n_qubits), 2))
    if readout_mode == "all_triples":
        return singles + list(combinations(range(n_qubits), 2)) + list(combinations(range(n_qubits), 3))
    if readout_mode == "all_quads":
        return (
            singles
            + list(combinations(range(n_qubits), 2))
            + list(combinations(range(n_qubits), 3))
            + list(combinations(range(n_qubits), 4))
        )
    raise ValueError(f"Unsupported readout_mode: {readout_mode}")


def quantum_readout_dim(n_qubits: int, readout_mode: str = "adjacent") -> int:
    return len(quantum_readout_terms(n_qubits, readout_mode))


def pauli_z_product(wires: Tuple[int, ...]):
    obs = qml.PauliZ(wires[0])
    for wire in wires[1:]:
        obs = obs @ qml.PauliZ(wire)
    return obs


class QuantumModule(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        q_layers: int,
        reuploads: int,
        q_feature_dim: int = 16,
        readout_mode: str = "adjacent",
        q_init_scale: float = 0.05,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.readout_mode = readout_mode
        self.readout_terms = quantum_readout_terms(n_qubits, readout_mode)

        self.dev = qml.device("default.qubit", wires=n_qubits)
        weight_shapes = {"weights": (reuploads, q_layers, n_qubits, 3)}

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for r in range(reuploads):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[r], wires=range(n_qubits))

            return [qml.expval(pauli_z_product(term)) for term in self.readout_terms]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        if q_init_scale > 0:
            with torch.no_grad():
                for p in self.qlayer.parameters():
                    p.uniform_(-q_init_scale, q_init_scale)

        self.readout = nn.Sequential(
            nn.Linear(quantum_readout_dim(n_qubits, readout_mode), q_feature_dim),
            nn.GELU(),
            nn.LayerNorm(q_feature_dim),
            nn.Dropout(0.10),
        )

    def forward(self, x):
        q_outs = [self.qlayer(x[i]) for i in range(x.shape[0])]
        return self.readout(torch.stack(q_outs, dim=0))


# =========================================================
# 4. Models
# =========================================================
def group_norm_for(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class HybridFeatureExtractor(nn.Module):
    """CNN front-end with separate classical and quantum projections."""

    def __init__(
        self,
        n_qubits: int,
        latent_dim: int = 8,
        classical_feature_dim: int = 16,
        conv_channels: Tuple[int, int] = (16, 32),
        bottleneck_hidden: int = 64,
        q_input_scale: float = math.pi,
        in_channels: int = 1,
    ):
        super().__init__()
        self.q_input_scale = q_input_scale
        c1, c2 = conv_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            group_norm_for(c1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            group_norm_for(c2),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(c2 * 2 * 2, bottleneck_hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(bottleneck_hidden, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.classical_proj = nn.Sequential(
            nn.Linear(latent_dim, classical_feature_dim),
            nn.GELU(),
            nn.LayerNorm(classical_feature_dim),
        )
        self.quantum_input_proj = nn.Sequential(
            nn.Linear(latent_dim, n_qubits),
            nn.LayerNorm(n_qubits),
        )

    def forward(self, x):
        h = self.bottleneck(self.conv(x))
        x_classical = self.classical_proj(h)
        x_quantum = self.q_input_scale * torch.tanh(self.quantum_input_proj(h))
        return x_quantum, x_classical


class OriginalHybridFeatureExtractor(nn.Module):
    """CNN front-end with shared n_qubits bottleneck."""

    def __init__(
        self,
        n_qubits: int,
        classical_feature_dim: int = 16,
        conv_channels: Tuple[int, int] = (16, 32),
        bottleneck_hidden: int = 64,
        q_input_scale: float = math.pi,
        in_channels: int = 1,
    ):
        super().__init__()
        self.q_input_scale = q_input_scale
        c1, c2 = conv_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            group_norm_for(c1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            group_norm_for(c2),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(c2 * 2 * 2, bottleneck_hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(bottleneck_hidden, n_qubits),
            nn.LayerNorm(n_qubits),
        )
        self.classical_proj = nn.Sequential(
            nn.Linear(n_qubits, classical_feature_dim),
            nn.GELU(),
            nn.LayerNorm(classical_feature_dim),
        )

    def forward(self, x):
        h = self.bottleneck(self.conv(x))
        x_quantum = self.q_input_scale * torch.tanh(h)
        x_classical = self.classical_proj(h)
        return x_quantum, x_classical


class HybridCNNQNN(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        q_layers: int,
        reuploads: int,
        num_classes: int,
        hybrid_style: str = "original",
        latent_dim: int = 8,
        classical_feature_dim: int = 16,
        quantum_feature_dim: int = 16,
        classifier_hidden: int = 32,
        conv_channels: Tuple[int, int] = (16, 32),
        bottleneck_hidden: int = 64,
        readout_mode: str = "adjacent",
        q_init_scale: float = 0.03,
        q_input_scale: float = math.pi,
        use_q_gate: bool = True,
        q_gate_init: float = 0.25,
        fusion_mode: str = "scalar",
        gate_hidden: int = 8,
        gate_dropout: float = 0.05,
        lambda_max: float = 0.50,
        lambda_init: float = 0.05,
        zero_init_quantum_head: bool = False,
        center_quantum_logits: bool = True,
        in_channels: int = 1,
    ):
        super().__init__()
        self.quantum_feature_dim = quantum_feature_dim
        self.classical_feature_dim = classical_feature_dim
        self.use_q_gate = use_q_gate
        self.hybrid_style = hybrid_style
        self.fusion_mode = fusion_mode
        self.lambda_max = float(lambda_max)
        self.center_quantum_logits = bool(center_quantum_logits)

        if hybrid_style == "original":
            self.frontend = OriginalHybridFeatureExtractor(
                n_qubits=n_qubits,
                classical_feature_dim=classical_feature_dim,
                conv_channels=conv_channels,
                bottleneck_hidden=bottleneck_hidden,
                q_input_scale=q_input_scale,
                in_channels=in_channels,
            )
        elif hybrid_style == "split":
            self.frontend = HybridFeatureExtractor(
                n_qubits=n_qubits,
                latent_dim=latent_dim,
                classical_feature_dim=classical_feature_dim,
                conv_channels=conv_channels,
                bottleneck_hidden=bottleneck_hidden,
                q_input_scale=q_input_scale,
                in_channels=in_channels,
            )
        else:
            raise ValueError(f"Unsupported hybrid_style: {hybrid_style}")

        self.qmodule = QuantumModule(
            n_qubits=n_qubits,
            q_layers=q_layers,
            reuploads=reuploads,
            q_feature_dim=quantum_feature_dim,
            readout_mode=readout_mode,
            q_init_scale=q_init_scale,
        )
        self.q_norm = nn.LayerNorm(quantum_feature_dim)

        if fusion_mode == "scalar":
            if use_q_gate:
                self.q_gate_logit = nn.Parameter(
                    torch.tensor(safe_logit(q_gate_init), dtype=torch.float32)
                )
            else:
                self.register_parameter("q_gate_logit", None)
            self.gate_net = None
            self.fusion_norm = nn.LayerNorm(classical_feature_dim + quantum_feature_dim)
            self.classifier = nn.Sequential(
                nn.Linear(classical_feature_dim + quantum_feature_dim, classifier_hidden),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(classifier_hidden, num_classes),
            )
            self.classical_head = None
            self.quantum_head = None

        elif fusion_mode == "dynamic":
            self.register_parameter("q_gate_logit", None)
            self.gate_net = nn.Sequential(
                nn.Linear(classical_feature_dim + quantum_feature_dim, gate_hidden),
                nn.GELU(),
                nn.Dropout(gate_dropout),
                nn.Linear(gate_hidden, quantum_feature_dim),
                nn.Sigmoid(),
            )
            with torch.no_grad():
                last_linear = self.gate_net[3]
                nn.init.zeros_(last_linear.weight)
                last_linear.bias.fill_(safe_logit(q_gate_init))
            self.fusion_norm = nn.LayerNorm(classical_feature_dim + quantum_feature_dim)
            self.classifier = nn.Sequential(
                nn.Linear(classical_feature_dim + quantum_feature_dim, classifier_hidden),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(classifier_hidden, num_classes),
            )
            self.classical_head = None
            self.quantum_head = None

        elif fusion_mode in ("logit_residual", "logit_residual_gate"):
            # The classical branch provides the main prediction; the quantum branch
            # learns a bounded residual correction in logit space.
            # logit_residual      : one global learnable lambda.
            # logit_residual_gate : sample-wise class-wise bounded lambda from [c, q].
            self.register_parameter("q_gate_logit", None)
            self.gate_net = None
            self.fusion_norm = None
            self.classifier = None
            self.classical_head = nn.Sequential(
                nn.Linear(classical_feature_dim, classifier_hidden),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(classifier_hidden, num_classes),
            )
            self.quantum_head = nn.Sequential(
                nn.Linear(quantum_feature_dim, classifier_hidden),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(classifier_hidden, num_classes),
            )
            if zero_init_quantum_head:
                with torch.no_grad():
                    last = self.quantum_head[-1]
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)

            if fusion_mode == "logit_residual":
                self.lambda_logit = nn.Parameter(
                    torch.tensor(safe_logit(lambda_init / max(lambda_max, 1e-8)), dtype=torch.float32)
                )
                self.logit_gate_net = None
            else:
                self.register_parameter("lambda_logit", None)
                self.logit_gate_net = nn.Sequential(
                    nn.Linear(classical_feature_dim + quantum_feature_dim, gate_hidden),
                    nn.GELU(),
                    nn.Dropout(gate_dropout),
                    nn.Linear(gate_hidden, num_classes),
                    nn.Sigmoid(),
                )
                # Start from a small quantum residual contribution.
                with torch.no_grad():
                    last = self.logit_gate_net[-2]
                    nn.init.zeros_(last.weight)
                    last.bias.fill_(safe_logit(lambda_init / max(lambda_max, 1e-8)))

        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    def quantum_lambda(self):
        if self.fusion_mode != "logit_residual":
            return None
        return self.lambda_max * torch.sigmoid(self.lambda_logit)

    def forward(self, x, quantum_scale: float = 1.0, return_parts: bool = False):
        x_for_quantum, x_classical = self.frontend(x)

        # Logit-residual modes: z = z_c + lambda * z_q.
        # This keeps the classical branch as a stable backbone and lets
        # the quantum branch provide a bounded residual correction.
        if self.fusion_mode in ("logit_residual", "logit_residual_gate"):
            z_classical = self.classical_head(x_classical)
            if quantum_scale == 0.0:
                z_quantum = torch.zeros_like(z_classical)
                logits = z_classical
            else:
                x_quantum = self.qmodule(x_for_quantum)
                x_quantum = self.q_norm(x_quantum)
                z_quantum = self.quantum_head(x_quantum)
                if self.center_quantum_logits:
                    z_quantum = z_quantum - z_quantum.mean(dim=1, keepdim=True)
                if self.fusion_mode == "logit_residual":
                    lam = self.quantum_lambda()
                else:
                    gate_input = torch.cat([x_classical, x_quantum], dim=1)
                    lam = self.lambda_max * self.logit_gate_net(gate_input)
                logits = z_classical + (quantum_scale * lam) * z_quantum
            if return_parts:
                return logits, z_classical, z_quantum
            return logits

        # Original feature-level fusion modes.
        if quantum_scale == 0.0:
            x_quantum = torch.zeros(
                x_classical.shape[0],
                self.quantum_feature_dim,
                device=x_classical.device,
                dtype=x_classical.dtype,
            )
        else:
            x_quantum = self.qmodule(x_for_quantum)
            x_quantum = self.q_norm(x_quantum)

            if self.use_q_gate:
                if self.fusion_mode == "scalar":
                    alpha = torch.sigmoid(self.q_gate_logit)
                    x_quantum = alpha * x_quantum
                elif self.fusion_mode == "dynamic":
                    gate_input = torch.cat([x_classical, x_quantum], dim=1)
                    gate = self.gate_net(gate_input)
                    x_quantum = gate * x_quantum

            if quantum_scale != 1.0:
                x_quantum = quantum_scale * x_quantum

        fused = torch.cat([x_classical, x_quantum], dim=1)
        fused = self.fusion_norm(fused)
        logits = self.classifier(fused)
        if return_parts:
            return logits, None, None
        return logits


class CNNControl(nn.Module):
    """Parameter-matched CNN using the same convolutional front-end."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 20,
        hidden_dim: int = 32,
        conv_channels: Tuple[int, int] = (16, 32),
        bottleneck_hidden: int = 64,
        in_channels: int = 1,
    ):
        super().__init__()
        c1, c2 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            group_norm_for(c1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            group_norm_for(c2),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(c2 * 2 * 2, bottleneck_hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(bottleneck_hidden, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, quantum_scale: float = 1.0):
        return self.classifier(self.conv(x))


class CNNFull(nn.Module):
    """Stronger CNN reference, not parameter-matched."""

    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(48 * 2 * 2, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 32),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, quantum_scale: float = 1.0):
        return self.classifier(self.conv(x))


class VGGSmallBaseline(nn.Module):
    """Compact VGG-style classical CNN baseline."""

    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            group_norm_for(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            group_norm_for(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            group_norm_for(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            group_norm_for(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, quantum_scale: float = 1.0):
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = group_norm_for(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = group_norm_for(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                group_norm_for(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class ResNetSmallBaseline(nn.Module):
    """Small ResNet-style classical CNN baseline."""

    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            group_norm_for(16),
            nn.GELU(),
        )
        self.features = nn.Sequential(
            ResidualBlock(16, 16, stride=1),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 48, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(48, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, num_classes),
        )

    def forward(self, x, quantum_scale: float = 1.0):
        return self.classifier(self.features(self.stem(x)))


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, drop_rate: float = 0.05):
        super().__init__()
        inter_channels = 4 * growth_rate
        self.net = nn.Sequential(
            group_norm_for(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            group_norm_for(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout2d(drop_rate),
        )

    def forward(self, x):
        return torch.cat([x, self.net(x)], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, growth_rate: int, drop_rate: float = 0.05):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate, drop_rate=drop_rate))
            channels += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.layers(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            group_norm_for(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class DenseNetSmallBaseline(nn.Module):
    """Small DenseNet-style classical CNN baseline."""

    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            group_norm_for(16),
            nn.GELU(),
        )
        block1 = DenseBlock(16, num_layers=3, growth_rate=8, drop_rate=0.05)
        trans1 = TransitionLayer(block1.out_channels, 24)
        block2 = DenseBlock(24, num_layers=3, growth_rate=8, drop_rate=0.05)
        trans2 = TransitionLayer(block2.out_channels, 32)
        block3 = DenseBlock(32, num_layers=3, growth_rate=8, drop_rate=0.05)
        self.features = nn.Sequential(block1, trans1, block2, trans2, block3)
        self.norm = group_norm_for(block3.out_channels)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(block3.out_channels, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, num_classes),
        )

    def forward(self, x, quantum_scale: float = 1.0):
        x = self.stem(x)
        x = self.norm(self.features(x))
        return self.classifier(x)


class LeNet5Baseline(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x, quantum_scale: float = 1.0):
        return self.classifier(self.features(x))


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])


# =========================================================
# 5. Build / train / eval
# =========================================================
def build_model(args):
    num_classes = len(args.class_ids)
    in_channels = get_in_channels(args.dataset)

    if args.model == "hybrid":
        return HybridCNNQNN(
            n_qubits=args.n_qubits,
            q_layers=args.q_layers,
            reuploads=args.reuploads,
            num_classes=num_classes,
            hybrid_style=args.hybrid_style,
            latent_dim=args.hybrid_latent_dim,
            classical_feature_dim=args.classical_dim,
            quantum_feature_dim=args.quantum_dim,
            classifier_hidden=args.classifier_hidden,
            conv_channels=tuple(args.conv_channels),
            bottleneck_hidden=args.bottleneck_hidden,
            readout_mode=args.readout_mode,
            q_init_scale=args.q_init_scale,
            q_input_scale=args.q_input_scale,
            use_q_gate=not args.no_q_gate,
            q_gate_init=args.q_gate_init,
            fusion_mode=args.fusion_mode,
            gate_hidden=args.gate_hidden,
            gate_dropout=args.gate_dropout,
            lambda_max=args.logit_residual_lambda_max,
            lambda_init=args.logit_residual_lambda_init,
            zero_init_quantum_head=args.zero_init_quantum_head,
            center_quantum_logits=not args.no_center_quantum_logits,
            in_channels=in_channels,
        ).to(args.device)

    if args.model == "cnn_control":
        return CNNControl(
            num_classes=num_classes,
            feature_dim=args.cnn_feature_dim,
            hidden_dim=args.cnn_hidden_dim,
            conv_channels=tuple(args.conv_channels),
            bottleneck_hidden=args.bottleneck_hidden,
            in_channels=in_channels,
        ).to(args.device)

    if args.model == "cnn_full":
        return CNNFull(num_classes=num_classes, in_channels=in_channels).to(args.device)

    if args.model == "lenet":
        return LeNet5Baseline(num_classes=num_classes, in_channels=in_channels).to(args.device)

    if args.model == "vgg_small":
        return VGGSmallBaseline(num_classes=num_classes, in_channels=in_channels).to(args.device)

    if args.model == "resnet_small":
        return ResNetSmallBaseline(num_classes=num_classes, in_channels=in_channels).to(args.device)

    if args.model == "densenet_small":
        return DenseNetSmallBaseline(num_classes=num_classes, in_channels=in_channels).to(args.device)

    raise ValueError(f"Unsupported model: {args.model}")


CORE_MODELS = ["cnn_control", "hybrid", "cnn_full", "lenet"]
EXTRA_CLASSICAL_MODELS = ["vgg_small", "resnet_small", "densenet_small"]


def resolve_model_names(model_name: str) -> List[str]:
    if model_name == "all":
        return CORE_MODELS
    if model_name == "classical_extra":
        return EXTRA_CLASSICAL_MODELS
    if model_name == "all_plus_classical":
        return CORE_MODELS + EXTRA_CLASSICAL_MODELS
    return [model_name]


def build_optimizer(args, model: nn.Module):
    classical_params = []
    quantum_circuit_params = []
    quantum_head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "qmodule.qlayer" in name:
            quantum_circuit_params.append(p)
        elif any(
            key in name
            for key in (
                "qmodule.readout",
                "q_norm",
                "quantum_head",
                "logit_gate_net",
                "lambda_logit",
                "q_gate_logit",
                "gate_net",
            )
        ):
            quantum_head_params.append(p)
        else:
            classical_params.append(p)

    param_groups = []
    if classical_params:
        param_groups.append({"params": classical_params, "lr": args.lr_classical, "name": "classical"})
    if quantum_circuit_params:
        param_groups.append({"params": quantum_circuit_params, "lr": args.lr_quantum, "name": "q_circuit"})
    if quantum_head_params:
        param_groups.append({"params": quantum_head_params, "lr": args.lr_quantum_head, "name": "q_head"})

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=args.min_lr,
    )
    return optimizer, scheduler


def set_quantum_trainable(model: nn.Module, trainable: bool) -> None:
    if not hasattr(model, "qmodule"):
        return
    for name, p in model.named_parameters():
        if "qmodule" in name:
            p.requires_grad_(trainable)


def get_quantum_scale(epoch: int, args) -> float:
    """
    Control the quantum branch participation during early training.
    off    : quantum branch is disabled during warmup.
    freeze : quantum branch is used but quantum parameters are frozen during warmup.
    linear : quantum branch contribution increases linearly during warmup.
    none   : full quantum branch is used from the first epoch.
    """
    if args.warmup_epochs <= 0 or args.warmup_mode == "none":
        return 1.0

    if args.warmup_mode == "off":
        return 0.0 if epoch <= args.warmup_epochs else 1.0

    if args.warmup_mode == "freeze":
        return 1.0

    if args.warmup_mode == "linear":
        if epoch <= args.warmup_epochs:
            return max(0.1, epoch / float(args.warmup_epochs))
        return 1.0

    return 1.0


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    grad_clip=1.0,
    freeze_quantum=False,
    quantum_scale=1.0,
    ema: Optional[ModelEMA] = None,
    aux_classical_loss_weight: float = 0.0,
    aux_quantum_loss_weight: float = 0.0,
):
    model.train()
    if freeze_quantum and hasattr(model, "qmodule"):
        model.qmodule.eval()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if (
            (aux_classical_loss_weight > 0 or aux_quantum_loss_weight > 0)
            and getattr(model, "fusion_mode", None) in ("logit_residual", "logit_residual_gate")
        ):
            logits, z_classical, z_quantum = model(x, quantum_scale=quantum_scale, return_parts=True)
            loss = criterion(logits, y)
            if aux_classical_loss_weight > 0:
                loss = loss + aux_classical_loss_weight * criterion(z_classical, y)
            if aux_quantum_loss_weight > 0 and z_quantum is not None and z_quantum.requires_grad:
                loss = loss + aux_quantum_loss_weight * criterion(z_quantum, y)
        else:
            logits = model(x, quantum_scale=quantum_scale)
            loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_num += y.size(0)
        total_loss += loss.item() * y.size(0)

    return total_loss / total_num, total_correct / total_num


def score_metrics(metrics, mode: str = "balanced", auc_weight: float = 0.30):
    auc = 0.0 if np.isnan(metrics["auc"]) else metrics["auc"]

    if mode == "f1":
        return metrics["f1"]
    if mode == "f1_auc":
        return metrics["f1"] + auc_weight * auc
    if mode == "auc":
        return auc
    if mode == "acc_f1":
        return 0.5 * metrics["acc"] + 0.5 * metrics["f1"]
    if mode == "balanced":
        return 0.40 * metrics["f1"] + 0.40 * auc + 0.20 * metrics["acc"]

    raise ValueError(f"Unsupported score_mode: {mode}")


@torch.no_grad()
def evaluate(model, loader, device, num_classes, criterion=None, quantum_scale=1.0):
    model.eval()
    all_targets = []
    all_probs = []
    total_loss = 0.0
    total_num = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x, quantum_scale=quantum_scale)
        probs = torch.softmax(logits, dim=1)

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_num += y.size(0)

        all_targets.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.argmax(all_probs, axis=1)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            y_onehot = np.eye(num_classes)[all_targets]
            auc = roc_auc_score(y_onehot, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    out = {"acc": acc, "f1": f1, "auc": auc}
    if criterion is not None and total_num > 0:
        out["loss"] = total_loss / total_num
    return out


def run_one_seed(args, seed: int):
    set_seed(seed)
    train_loader, val_loader, test_loader = build_loaders(args, seed=seed)
    model = build_model(args)
    optimizer, scheduler = build_optimizer(args, model)
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    params = count_parameters(model)
    print("\n" + "=" * 72)
    print(f"[{args.model}] seed={seed} dataset={args.dataset} classes={tuple(args.class_ids)} k={args.k_shot}")
    print(f"params={params} device={args.device}")
    print("=" * 72)

    best_score = -1.0
    best_metrics = None
    best_state = None
    best_epoch = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        has_quantum = hasattr(model, "qmodule")
        in_warmup = has_quantum and epoch <= args.warmup_epochs and args.warmup_mode != "none"

        freeze_quantum = in_warmup and args.warmup_mode == "freeze"
        quantum_scale = get_quantum_scale(epoch, args)
        set_quantum_trainable(model, not freeze_quantum)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=train_criterion,
            device=args.device,
            grad_clip=args.grad_clip,
            freeze_quantum=freeze_quantum,
            quantum_scale=quantum_scale,
            ema=ema,
            aux_classical_loss_weight=args.aux_classical_loss_weight,
            aux_quantum_loss_weight=args.aux_quantum_loss_weight,
        )

        eval_model = ema.ema if ema is not None else model

        val_quantum_scale = 1.0
        if in_warmup and args.warmup_mode == "off" and not args.warmup_eval_full:
            val_quantum_scale = 0.0

        val_metrics = evaluate(
            eval_model,
            val_loader,
            args.device,
            len(args.class_ids),
            criterion=eval_criterion,
            quantum_scale=val_quantum_scale,
        )
        score = score_metrics(val_metrics, mode=args.score_mode, auc_weight=args.auc_score_weight)

        # When quantum branch is fully off, do not save the warmup-only model as the best model.
        skip_best_update = in_warmup and args.warmup_mode == "off"

        if not skip_best_update:
            scheduler.step(val_metrics["loss"])
            if score > best_score:
                best_score = score
                best_metrics = copy.deepcopy(val_metrics)
                best_state = copy.deepcopy(eval_model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

        if has_quantum and args.warmup_mode == "linear" and in_warmup:
            phase = f"[Q-linear:{quantum_scale:.2f}]"
        elif quantum_scale == 0.0:
            phase = "[warmup-off]"
        elif freeze_quantum:
            phase = "[Q-freeze]"
        else:
            phase = "[full]"

        lr_text = " ".join(
            f"lr_{pg.get('name', i)}={pg['lr']:.2e}"
            for i, pg in enumerate(optimizer.param_groups)
        )
        print(
            f"Epoch [{epoch:02d}/{args.epochs}] {phase} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"score={score:.4f} {lr_text}"
        )

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best model was saved.")

    eval_model = ema.ema if ema is not None else model
    eval_model.load_state_dict(best_state)

    test_metrics = evaluate(
        eval_model,
        test_loader,
        args.device,
        len(args.class_ids),
        criterion=eval_criterion,
        quantum_scale=1.0,
    )

    print("Best val metrics:", best_metrics)
    print("Best epoch:", best_epoch, "Best score:", best_score)
    print("Final test metrics:", test_metrics)

    return {
        "seed": seed,
        "model": args.model,
        "k_shot": args.k_shot,
        "val_shot": args.val_shot,
        "params": params,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val": best_metrics,
        "test": test_metrics,
    }


def summarize_runs(runs: List[dict]):
    if not runs:
        return {
            "num_runs": 0,
            "params": None,
            "val": {
                "acc_mean": float("nan"),
                "acc_std": float("nan"),
                "f1_mean": float("nan"),
                "f1_std": float("nan"),
                "auc_mean": float("nan"),
                "auc_std": float("nan"),
            },
            "test": {
                "acc_mean": float("nan"),
                "acc_std": float("nan"),
                "f1_mean": float("nan"),
                "f1_std": float("nan"),
                "auc_mean": float("nan"),
                "auc_std": float("nan"),
            },
        }

    val_accs = [r["best_val"]["acc"] for r in runs]
    val_f1s = [r["best_val"]["f1"] for r in runs]
    val_aucs = [r["best_val"]["auc"] for r in runs]
    test_accs = [r["test"]["acc"] for r in runs]
    test_f1s = [r["test"]["f1"] for r in runs]
    test_aucs = [r["test"]["auc"] for r in runs]

    return {
        "num_runs": len(runs),
        "params": runs[0]["params"] if runs else None,
        "val": {
            "acc_mean": safe_mean(val_accs),
            "acc_std": safe_std(val_accs),
            "f1_mean": safe_mean(val_f1s),
            "f1_std": safe_std(val_f1s),
            "auc_mean": safe_mean(val_aucs),
            "auc_std": safe_std(val_aucs),
        },
        "test": {
            "acc_mean": safe_mean(test_accs),
            "acc_std": safe_std(test_accs),
            "f1_mean": safe_mean(test_f1s),
            "f1_std": safe_std(test_f1s),
            "auc_mean": safe_mean(test_aucs),
            "auc_std": safe_std(test_aucs),
        },
    }


# =========================================================
# 6. Main
# =========================================================
def cli_arg_was_supplied(name: str) -> bool:
    dashed = "--" + name.replace("_", "-")
    underscored = "--" + name
    neg_dashed = "--no-" + name.replace("_", "-")
    neg_underscored = "--no_" + name
    candidates = (dashed, underscored, neg_dashed, neg_underscored)
    for token in sys.argv[1:]:
        if token in candidates:
            return True
        if any(token.startswith(c + "=") for c in candidates):
            return True
    return False


def apply_mnist_preset(args):
    if args.preset == "none":
        return args

    presets = {
        "mnist35": {
            "dataset": "MNIST",
            "class_ids": [3, 5],
            "k_shots": [10, 20, 50, 100],
            "val_shot": 100,
            "batch_size": 16,
            "epochs": 50,
            "patience": 14,
            "n_qubits": 6,
            "q_layers": 2,
            "reuploads": 1,
            "readout_mode": "all_pairs",
            "q_init_scale": 0.025,
            "q_input_scale": math.pi,
            "q_gate_init": 0.25,
            "fusion_mode": "logit_residual_gate",
            "logit_residual_lambda_max": 0.40,
            "logit_residual_lambda_init": 0.03,
            "aux_classical_loss_weight": 0.10,
            "aux_quantum_loss_weight": 0.03,
            "zero_init_quantum_head": True,
            "gate_hidden": 8,
            "gate_dropout": 0.05,
            "warmup_mode": "linear",
            "warmup_epochs": 5,
            "hybrid_style": "split",
            "hybrid_latent_dim": 8,
            "classical_dim": 16,
            "quantum_dim": 16,
            "classifier_hidden": 32,
            "conv_channels": [16, 32],
            "bottleneck_hidden": 64,
            "cnn_feature_dim": 27,
            "cnn_hidden_dim": 32,
            "lr_classical": 1e-3,
            "lr_quantum": 5e-5,
            "lr_quantum_head": 6e-4,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "label_smoothing": 0.01,
            "aug_mode": "medium",
            "score_mode": "acc_f1",
            "seeds": [0, 1, 2, 3, 4],
            "summary_path": "results_mnist35_qrf_v7_5seeds_summary.json",
        },
        "mnist35_stable": {
            "dataset": "MNIST",
            "class_ids": [3, 5],
            "k_shots": [10, 20, 50, 100],
            "val_shot": 100,
            "batch_size": 16,
            "epochs": 60,
            "patience": 18,
            "n_qubits": 6,
            "q_layers": 2,
            "reuploads": 2,
            "readout_mode": "all_pairs",
            "q_init_scale": 0.020,
            "q_input_scale": math.pi,
            "q_gate_init": 0.18,
            "fusion_mode": "logit_residual_gate",
            "logit_residual_lambda_max": 0.30,
            "logit_residual_lambda_init": 0.02,
            "aux_classical_loss_weight": 0.10,
            "aux_quantum_loss_weight": 0.02,
            "zero_init_quantum_head": True,
            "gate_hidden": 8,
            "gate_dropout": 0.03,
            "warmup_mode": "linear",
            "warmup_epochs": 8,
            "hybrid_style": "split",
            "hybrid_latent_dim": 8,
            "classical_dim": 16,
            "quantum_dim": 16,
            "classifier_hidden": 32,
            "conv_channels": [16, 32],
            "bottleneck_hidden": 64,
            "cnn_feature_dim": 27,
            "cnn_hidden_dim": 32,
            "lr_classical": 1e-3,
            "lr_quantum": 4e-5,
            "lr_quantum_head": 5e-4,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "label_smoothing": 0.01,
            "aug_mode": "medium",
            "use_ema": True,
            "ema_decay": 0.995,
            "score_mode": "acc_f1",
            "seeds": [0, 1, 2, 3, 4],
            "summary_path": "results_mnist35_qrf_v7_5seeds_summary.json",
        },
        "mnist358": {
            "dataset": "MNIST",
            "class_ids": [3, 5, 8],
            "k_shots": [10, 20, 50, 100],
            "val_shot": 100,
            "batch_size": 16,
            "epochs": 55,
            "patience": 15,
            "n_qubits": 6,
            "q_layers": 2,
            "reuploads": 1,
            "readout_mode": "all_pairs",
            "q_init_scale": 0.03,
            "q_input_scale": math.pi,
            "q_gate_init": 0.25,
            "fusion_mode": "logit_residual_gate",
            "logit_residual_lambda_max": 0.45,
            "logit_residual_lambda_init": 0.04,
            "aux_classical_loss_weight": 0.10,
            "aux_quantum_loss_weight": 0.05,
            "zero_init_quantum_head": True,
            "gate_hidden": 8,
            "gate_dropout": 0.05,
            "warmup_mode": "linear",
            "warmup_epochs": 6,
            "hybrid_style": "split",
            "hybrid_latent_dim": 8,
            "classical_dim": 16,
            "quantum_dim": 16,
            "classifier_hidden": 32,
            "conv_channels": [16, 32],
            "bottleneck_hidden": 64,
            "cnn_feature_dim": 27,
            "cnn_hidden_dim": 32,
            "lr_classical": 1e-3,
            "lr_quantum": 5e-5,
            "lr_quantum_head": 5e-4,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "label_smoothing": 0.01,
            "aug_mode": "medium",
            "score_mode": "acc_f1",
            "seeds": [0, 1, 2, 3, 4],
            "summary_path": "results_mnist358_qrf_v7_5seeds_summary.json",
        },
        "mnist358_stable": {
            "dataset": "MNIST",
            "class_ids": [3, 5, 8],
            "k_shots": [10, 50, 100],
            "val_shot": 100,
            "batch_size": 16,
            "epochs": 65,
            "patience": 18,
            "n_qubits": 6,
            "q_layers": 2,
            "reuploads": 2,
            "readout_mode": "all_pairs",
            "q_init_scale": 0.025,
            "q_input_scale": math.pi,
            "q_gate_init": 0.20,
            "fusion_mode": "logit_residual_gate",
            "logit_residual_lambda_max": 0.38,
            "logit_residual_lambda_init": 0.03,
            "aux_classical_loss_weight": 0.10,
            "aux_quantum_loss_weight": 0.03,
            "zero_init_quantum_head": True,
            "gate_hidden": 8,
            "gate_dropout": 0.03,
            "warmup_mode": "linear",
            "warmup_epochs": 8,
            "hybrid_style": "split",
            "hybrid_latent_dim": 8,
            "classical_dim": 16,
            "quantum_dim": 16,
            "classifier_hidden": 32,
            "conv_channels": [16, 32],
            "bottleneck_hidden": 64,
            "cnn_feature_dim": 27,
            "cnn_hidden_dim": 32,
            "lr_classical": 1e-3,
            "lr_quantum": 4e-5,
            "lr_quantum_head": 5e-4,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "label_smoothing": 0.01,
            "aug_mode": "medium",
            "use_ema": True,
            "ema_decay": 0.995,
            "score_mode": "acc_f1",
            "seeds": [0, 1, 2, 3, 4],
            "summary_path": "results_mnist358_qrf_v7_5seeds_summary.json",
        },
    }

    for key, value in presets[args.preset].items():
        if not cli_arg_was_supplied(key):
            setattr(args, key, copy.deepcopy(value))
    return args


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument("--preset", type=str, default="mnist35_stable",
                        choices=["mnist35", "mnist35_stable", "mnist358", "mnist358_stable", "none"],
                        help="MNIST preset. Use none to rely only on explicit CLI arguments.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=[
            "hybrid",
            "cnn_control",
            "cnn_full",
            "lenet",
            "vgg_small",
            "resnet_small",
            "densenet_small",
            "all",
            "classical_extra",
            "all_plus_classical",
        ],
    )
    parser.add_argument("--dataset", type=str, default="FashionMNIST",
                        choices=["MNIST", "FashionMNIST", "CIFAR10"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--class_ids", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--k_shot", type=int, default=20)
    parser.add_argument("--k_shots", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--val_shot", type=int, default=50,
                        help="-1 means validation shots per class equal current k_shot")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=12)

    # Hybrid/QRF-Net
    parser.add_argument("--n_qubits", type=int, default=6)
    parser.add_argument("--q_layers", type=int, default=2)
    parser.add_argument("--reuploads", type=int, default=1)
    parser.add_argument("--readout_mode", type=str, default="adjacent",
                        choices=["z", "adjacent", "all_pairs", "all_triples", "all_quads"])
    parser.add_argument("--q_init_scale", type=float, default=0.03)
    parser.add_argument("--q_input_scale", type=float, default=math.pi)
    parser.add_argument("--no_q_gate", action="store_true")
    parser.add_argument("--q_gate_init", type=float, default=0.25)
    parser.add_argument("--fusion_mode", type=str, default="logit_residual_gate",
                        choices=["scalar", "dynamic", "logit_residual", "logit_residual_gate"])
    parser.add_argument("--logit_residual_lambda_max", type=float, default=0.45,
                        help="Upper bound of the learnable quantum residual coefficient.")
    parser.add_argument("--logit_residual_lambda_init", type=float, default=0.04,
                        help="Initial value of the quantum residual coefficient.")
    parser.add_argument("--aux_classical_loss_weight", type=float, default=0.10,
                        help="Auxiliary CE loss weight on the classical logits in logit residual modes.")
    parser.add_argument("--aux_quantum_loss_weight", type=float, default=0.05,
                        help="Auxiliary CE loss weight on the quantum residual logits in logit residual modes.")
    parser.add_argument("--zero_init_quantum_head", dest="zero_init_quantum_head",
                        action="store_true", default=True,
                        help="Zero-initialize the final quantum residual head for a safer residual start.")
    parser.add_argument("--no_zero_init_quantum_head", dest="zero_init_quantum_head",
                        action="store_false")
    parser.add_argument("--no_center_quantum_logits", action="store_true",
                        help="Disable per-sample centering of quantum residual logits before fusion.")
    parser.add_argument("--gate_hidden", type=int, default=8)
    parser.add_argument("--gate_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_mode", type=str, default="off",
                        choices=["off", "freeze", "none", "linear"])
    parser.add_argument("--warmup_epochs", type=int, default=4)
    parser.add_argument("--hybrid_style", type=str, default="original",
                        choices=["original", "split"])
    parser.add_argument("--hybrid_latent_dim", type=int, default=4)
    parser.add_argument("--classical_dim", type=int, default=16)
    parser.add_argument("--quantum_dim", type=int, default=16)
    parser.add_argument("--classifier_hidden", type=int, default=32)
    parser.add_argument("--conv_channels", type=int, nargs=2, default=[16, 32])
    parser.add_argument("--bottleneck_hidden", type=int, default=64)

    # CNN-control
    parser.add_argument("--cnn_feature_dim", type=int, default=20)
    parser.add_argument("--cnn_hidden_dim", type=int, default=32)

    # Optimization
    parser.add_argument("--lr_classical", type=float, default=1e-3)
    parser.add_argument("--lr_quantum", type=float, default=5e-5)
    parser.add_argument("--lr_quantum_head", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--aug_mode", type=str, default="light",
                        choices=["none", "light", "medium"])
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--warmup_eval_full", action="store_true",
                        help="Evaluate the full quantum branch during warmup-off epochs.")

    # Model selection
    parser.add_argument("--score_mode", type=str, default="balanced",
                        choices=["f1", "f1_auc", "auc", "acc_f1", "balanced"])
    parser.add_argument("--auc_score_weight", type=float, default=0.30)

    # Runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(20)))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--loader_seed_offset", type=int, default=100000)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--summary_path", type=str, default="results_mnist_qrf_v7_5seeds_summary.json")
    parser.add_argument(
        "--save_all_runs",
        action="store_true",
        help="Save all repeated random-seed raw runs in JSON. By default a compact selected-seed summary is saved.",
    )
    parser.add_argument("--print_params_only", action="store_true")

    return parser.parse_args()


def print_parameter_probe(args):
    device_bak = args.device
    model_bak = args.model

    args.device = "cpu"
    rows = []
    for model_name in resolve_model_names(args.model):
        args.model = model_name
        model = build_model(args)
        rows.append((model_name, count_parameters(model)))

    args.device = device_bak
    args.model = model_bak

    print("\nParameter probe")
    print("-" * 40)
    for name, params in rows:
        print(f"{name:12s}: {params:,}")


def requested_k_shots(args) -> List[int]:
    if args.k_shots is not None and len(args.k_shots) > 0:
        return [int(k) for k in args.k_shots]
    return [int(args.k_shot)]


def main():
    args = parse_args()
    args = apply_mnist_preset(args)

    if not cli_arg_was_supplied("summary_path") and args.model in {"classical_extra", "all_plus_classical"}:
        class_tag = "".join(str(c) for c in args.class_ids)
        args.summary_path = f"results_mnist{class_tag}_qrf_v7_{args.model}_5seeds_summary.json"

    if args.print_params_only:
        print_parameter_probe(args)
        return

    print("RUNNING FILE =", __file__)
    seeds = args.seeds if args.seeds is not None and len(args.seeds) > 0 else [args.seed]
    # All explicitly requested seeds are included in the reported mean/std.
    # No post-hoc seed ranking or metric-based seed filtering is performed.
    model_names = resolve_model_names(args.model)

    original_model = args.model
    original_k_shot = args.k_shot
    original_val_shot = args.val_shot
    k_shots = requested_k_shots(args)

    all_results = {}
    for k_shot in k_shots:
        args.k_shot = int(k_shot)
        args.val_shot = args.k_shot if original_val_shot < 0 else original_val_shot
        k_key = f"K={args.k_shot}"
        all_results[k_key] = {}

        for model_name in model_names:
            args.model = model_name
            print("\n" + "#" * 72)
            print(f"[RUN MODEL] {model_name} | {k_key} val_shot={args.val_shot}")
            print("#" * 72)

            runs = [run_one_seed(args, seed) for seed in seeds]
            all_results[k_key][model_name] = {
                "runs": runs,
                "summary": summarize_runs(runs),
            }

    args.model = original_model
    args.k_shot = original_k_shot
    args.val_shot = original_val_shot

    payload = {
        "config": vars(args),
    }
    if args.save_all_runs:
        payload["shots"] = all_results
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("[ALL-SHOT SUMMARY]")
    for k_key, by_model in all_results.items():
        print(f"\n{k_key}")
        for model_name in model_names:
            summary = by_model[model_name]["summary"]
            test = summary["test"]
            print(
                f"{model_name:12s} | Params={summary['params']:,} | "
                f"Test Acc={test['acc_mean']:.4f}+/-{test['acc_std']:.4f} | "
                f"F1={test['f1_mean']:.4f}+/-{test['f1_std']:.4f} | "
                f"AUC={test['auc_mean']:.4f}+/-{test['auc_std']:.4f}"
            )

    print(f"Saved summary to: {args.summary_path}")



if __name__ == "__main__":
    main()
