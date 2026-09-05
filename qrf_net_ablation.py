# -*- coding: utf-8 -*-
"""
QRF-Net V7 ablation runner.

This script reuses qrf_net_main.py and runs focused random-seed ablations.
It saves compact JSON summaries together with the completed seed runs.

Recommended paper-use ablations:
1) fusion: CNN-control vs dynamic feature fusion vs logit residual vs gated logit residual.
2) quantum: CNN-control vs z/all-pairs readout and 1/2 data re-uploads.
3) readout: Z, Z+adjacent ZZ, Z+all ZZ, plus higher-order ZZZ/ZZZZ readouts.
4) q_layers: quantum circuit depth ablation with all other QRF-Net settings fixed.
"""

import argparse
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "qrf_net_main.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("qrf_v7_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp = load_base_module()

RUNNER_VERSION = "2026-09-01-lab-stable"


PRESETS = {
    "fashion24": {
        "dataset": "FashionMNIST",
        "class_ids": [2, 4],
        "k_shots": [10, 20, 50, 100],
        "val_shot": 100,
        "epochs": 55,
        "patience": 16,
        "lambda_max": 0.35,
        "warmup_epochs": 6,
    },
    "fashion246": {
        "dataset": "FashionMNIST",
        "class_ids": [2, 4, 6],
        "k_shots": [10, 20, 50, 100],
        "val_shot": 100,
        "epochs": 60,
        "patience": 16,
        "lambda_max": 0.38,
        "warmup_epochs": 7,
    },
    "mnist35": {
        "dataset": "MNIST",
        "class_ids": [3, 5],
        "k_shots": [10, 20, 50, 100],
        "val_shot": 100,
        "epochs": 60,
        "patience": 16,
        "lambda_max": 0.35,
        "warmup_epochs": 7,
    },
    "mnist358": {
        "dataset": "MNIST",
        "class_ids": [3, 5, 8],
        "k_shots": [10, 20, 50, 100],
        "val_shot": 100,
        "epochs": 60,
        "patience": 16,
        "lambda_max": 0.35,
        "warmup_epochs": 7,
    },
}


FUSION_VARIANTS = [
    (
        "cnn_control",
        "CNN-control, no quantum branch",
        {
            "model": "cnn_control",
        },
    ),
    (
        "feature_dynamic",
        "Feature-level dynamic gated fusion",
        {
            "model": "hybrid",
            "fusion_mode": "dynamic",
            "aux_classical_loss_weight": 0.0,
            "aux_quantum_loss_weight": 0.0,
        },
    ),
    (
        "logit_residual",
        "Logit-level quantum residual",
        {
            "model": "hybrid",
            "fusion_mode": "logit_residual",
        },
    ),
    (
        "gated_logit_residual",
        "Class-wise gated logit-level quantum residual",
        {
            "model": "hybrid",
            "fusion_mode": "logit_residual_gate",
        },
    ),
]


QUANTUM_VARIANTS = [
    (
        "cnn_control",
        "CNN-control, no quantum branch",
        {
            "model": "cnn_control",
        },
    ),
    (
        "basic_z",
        "1 data re-upload, single-qubit Z readout",
        {
            "model": "hybrid",
            "reuploads": 1,
            "readout_mode": "z",
        },
    ),
    (
        "readout_all_pairs",
        "1 data re-upload, all-pairs readout",
        {
            "model": "hybrid",
            "reuploads": 1,
            "readout_mode": "all_pairs",
        },
    ),
    (
        "reupload_z",
        "2 data re-uploads, single-qubit Z readout",
        {
            "model": "hybrid",
            "reuploads": 2,
            "readout_mode": "z",
        },
    ),
    (
        "full_reupload_all_pairs",
        "2 data re-uploads, all-pairs readout",
        {
            "model": "hybrid",
            "reuploads": 2,
            "readout_mode": "all_pairs",
        },
    ),
]


READOUT_VARIANTS = [
    (
        "z",
        "Single-qubit Z readout",
        {
            "model": "hybrid",
            "reuploads": 2,
            "readout_mode": "z",
        },
    ),
    (
        "adjacent_zz",
        "Z + adjacent ZZ readout",
        {
            "model": "hybrid",
            "reuploads": 2,
            "readout_mode": "adjacent",
        },
    ),
    (
        "all_pairs",
        "Z + all two-body ZZ readout",
        {
            "model": "hybrid",
            "reuploads": 2,
            "readout_mode": "all_pairs",
        },
    ),
    (
        "all_triples",
        "Z + all ZZ + all three-body ZZZ readout",
        {
            "model": "hybrid",
            "reuploads": 2,
            "readout_mode": "all_triples",
        },
    ),
    (
        "all_quads",
        "Z + all ZZ + all ZZZ + all four-body ZZZZ readout",
        {
            "model": "hybrid",
            "reuploads": 2,
            "readout_mode": "all_quads",
        },
    ),
]


def build_q_layer_variants(q_layers_values: List[int]):
    return [
        (
            f"q_layers_{int(q_layers)}",
            f"{int(q_layers)} quantum circuit layer(s)",
            {
                "model": "hybrid",
                "q_layers": int(q_layers),
            },
        )
        for q_layers in q_layers_values
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="fashion246", choices=sorted(PRESETS))
    parser.add_argument("--ablation", type=str, default="fusion",
                        choices=["fusion", "quantum", "readout", "q_layers", "all"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--k_shots", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    parser.add_argument(
        "--seed_map_path",
        type=str,
        default=None,
        help="Optional previous-result JSON. Per-shot random-seed groups are loaded from it.",
    )
    parser.add_argument(
        "--readout_variants",
        type=str,
        nargs="*",
        default=None,
        choices=[name for name, _, _ in READOUT_VARIANTS],
        help="Optional subset for --ablation readout, e.g. all_pairs all_triples all_quads.",
    )
    parser.add_argument("--q_layers_values", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--force_quantum_cpu",
        action="store_true",
        help="Run PennyLane quantum circuit/readout on CPU while keeping other modules on --device.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing summary JSON and skip already completed seeds.",
    )
    parser.add_argument("--summary_path", type=str, default=None)
    return parser.parse_args()


def validate_cli_args(cli_args):
    if cli_args.ablation == "q_layers":
        if not cli_args.q_layers_values:
            raise ValueError("--q_layers_values requires at least one value.")
        bad_values = [x for x in cli_args.q_layers_values if int(x) < 1]
        if bad_values:
            raise ValueError(f"q_layers values must be >= 1, got: {bad_values}")
    if not cli_args.seeds:
        raise ValueError("--seeds requires at least one seed.")
    if cli_args.readout_variants is not None and len(cli_args.readout_variants) == 0:
        raise ValueError("--readout_variants was provided but no variant names were listed.")
    if cli_args.seed_map_path is not None and not Path(cli_args.seed_map_path).exists():
        raise FileNotFoundError(f"--seed_map_path does not exist: {cli_args.seed_map_path}")


def fresh_base_args():
    old_argv = sys.argv[:]
    sys.argv = [str(BASE_SCRIPT)]
    try:
        return exp.parse_args()
    finally:
        sys.argv = old_argv


def apply_common_config(base_args, cli_args):
    preset = PRESETS[cli_args.preset]
    base_args.dataset = preset["dataset"]
    base_args.class_ids = list(preset["class_ids"])
    base_args.k_shots = list(cli_args.k_shots) if cli_args.k_shots else list(preset["k_shots"])
    base_args.val_shot = int(preset["val_shot"])
    base_args.batch_size = 16
    base_args.epochs = int(preset["epochs"])
    base_args.patience = int(preset["patience"])
    base_args.data_root = cli_args.data_root

    base_args.n_qubits = 6
    base_args.q_layers = 2
    base_args.reuploads = 2
    base_args.readout_mode = "all_pairs"
    base_args.q_init_scale = 0.025
    base_args.q_input_scale = math.pi
    base_args.no_q_gate = False
    base_args.q_gate_init = 0.20
    base_args.fusion_mode = "logit_residual_gate"
    base_args.logit_residual_lambda_max = float(preset["lambda_max"])
    base_args.logit_residual_lambda_init = 0.03
    base_args.aux_classical_loss_weight = 0.10
    base_args.aux_quantum_loss_weight = 0.03
    base_args.zero_init_quantum_head = True
    base_args.no_center_quantum_logits = False
    base_args.gate_hidden = 8
    base_args.gate_dropout = 0.03
    base_args.warmup_mode = "linear"
    base_args.warmup_epochs = int(preset["warmup_epochs"])
    base_args.hybrid_style = "split"
    base_args.hybrid_latent_dim = 8
    base_args.classical_dim = 16
    base_args.quantum_dim = 16
    base_args.classifier_hidden = 32
    base_args.conv_channels = [16, 32]
    base_args.bottleneck_hidden = 64

    base_args.cnn_feature_dim = 27
    base_args.cnn_hidden_dim = 32

    base_args.lr_classical = 1e-3
    base_args.lr_quantum = 4e-5
    base_args.lr_quantum_head = 5e-4
    base_args.min_lr = 1e-5
    base_args.weight_decay = 1e-4
    base_args.grad_clip = 1.0
    base_args.label_smoothing = 0.01
    base_args.aug_mode = "medium"
    base_args.use_ema = True
    base_args.ema_decay = 0.995
    base_args.warmup_eval_full = False

    base_args.score_mode = "acc_f1"
    base_args.auc_score_weight = 0.30

    base_args.seed = 42
    base_args.seeds = list(cli_args.seeds)
    if cli_args.device is not None:
        base_args.device = cli_args.device
    base_args.force_quantum_cpu = bool(cli_args.force_quantum_cpu)
    base_args.num_workers = 0
    base_args.loader_seed_offset = 100000
    base_args.overfit_small = False
    base_args.print_params_only = False
    return base_args


def clone_args(args):
    return copy.deepcopy(args)


def compact_run(run: dict) -> dict:
    return {
        "seed": int(run["seed"]),
        "params": int(run["params"]),
        "best_epoch": int(run["best_epoch"]) if run.get("best_epoch") is not None else None,
        "best_val": run["best_val"],
        "test": run["test"],
    }


def build_variant_result(
    description: str,
    overrides: Dict,
    runs: List[dict],
    status: str = "complete",
    error: str = None,
) -> dict:
    """Summarize every completed seed without post-hoc filtering."""
    ordered_runs = sorted(runs, key=lambda r: int(r["seed"]))
    result = {
        "description": description,
        "overrides": overrides,
        "status": status,
        "summary": exp.summarize_runs(ordered_runs),
        "seeds": [int(r["seed"]) for r in ordered_runs],
        "runs": [compact_run(r) for r in ordered_runs],
        "completed_seeds": [int(r["seed"]) for r in ordered_runs],
        "_all_runs": ordered_runs,
    }
    if error is not None:
        result["error"] = error
    return result


def public_payload(payload: dict) -> dict:
    out = copy.deepcopy(payload)
    for group_result in out.get("groups", {}).values():
        strip_private_runs(group_result)
    return out


def save_payload(payload: dict, summary_path: Path):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(public_payload(payload), f, indent=2, ensure_ascii=False)


def load_completed_runs(existing_result: dict) -> List[dict]:
    if not isinstance(existing_result, dict):
        return []
    runs = existing_result.get("runs") or []
    if not isinstance(runs, list):
        return []

    deduped = {}
    for run in runs:
        if isinstance(run, dict) and "seed" in run and "test" in run and "best_val" in run:
            deduped[int(run["seed"])] = run
    return [deduped[seed] for seed in sorted(deduped)]


def parse_k_key(k_key) -> int:
    if isinstance(k_key, int):
        return int(k_key)
    text = str(k_key).strip()
    if "=" in text:
        text = text.split("=", 1)[1]
    return int(text)


def load_seed_map(seed_map_path: str) -> Dict[int, List[int]]:
    """Load the exact seed lists recorded in a previous summary.

    This function never ranks seeds by validation or test performance. It only
    reuses the seed identities that were explicitly run in the source summary.
    """
    path = Path(seed_map_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, List[int]] = {}

    shots = data.get("shots")
    if isinstance(shots, dict):
        for k_key, shot_result in shots.items():
            if not isinstance(shot_result, dict):
                continue
            model_block = (
                shot_result.get("hybrid")
                or shot_result.get("qrf")
                or shot_result.get("QRF")
            )
            runs = model_block.get("runs") if isinstance(model_block, dict) else None
            if runs:
                out[parse_k_key(k_key)] = [int(r["seed"]) for r in runs]

    if not out:
        groups = data.get("groups")
        if isinstance(groups, dict):
            for group in groups.values():
                if not isinstance(group, dict):
                    continue
                for k_key, shot_result in group.items():
                    if not isinstance(shot_result, dict):
                        continue
                    seeds = shot_result.get("seeds_run")
                    if seeds:
                        out[parse_k_key(k_key)] = [int(seed) for seed in seeds]

    if not out:
        raise ValueError(
            f"Cannot load explicit seed lists from {path}; expected shots/*/runs or groups/*/seeds_run."
        )
    return out


def clear_cuda_cache(args):
    try:
        if str(getattr(args, "device", "")).startswith("cuda") and exp.torch.cuda.is_available():
            exp.torch.cuda.empty_cache()
    except Exception:
        pass


def run_variant_for_shot(
    base_args,
    variant_name: str,
    description: str,
    overrides: Dict,
    k_shot: int,
    seeds: List[int],
    existing_result: dict = None,
    save_hook=None,
    resume: bool = False,
):
    args = clone_args(base_args)
    args.k_shot = int(k_shot)
    for key, value in overrides.items():
        setattr(args, key, copy.deepcopy(value))

    print("\n" + "=" * 80)
    print(f"[ABLATION VARIANT] {variant_name} | k={k_shot} | {description}")
    print("=" * 80)

    runs = load_completed_runs(existing_result) if resume else []
    done_seeds = {int(r["seed"]) for r in runs}
    if done_seeds:
        print(f"[RESUME] completed seeds for {variant_name} | k={k_shot}: {sorted(done_seeds)}")

    for seed in seeds:
        seed = int(seed)
        if resume and seed in done_seeds:
            print(f"[SKIP] {variant_name} | k={k_shot} | seed={seed} already completed")
            continue

        try:
            run = exp.run_one_seed(args, seed)
            runs.append(run)
            done_seeds.add(seed)
            result = build_variant_result(
                description=description,
                overrides=overrides,
                runs=runs,
                status="running" if len(done_seeds) < len(set(map(int, seeds))) else "complete",
            )
            if save_hook is not None:
                save_hook(result)
        except Exception as exc:
            result = build_variant_result(
                description=description,
                overrides=overrides,
                runs=runs,
                status="failed",
                error=repr(exc),
            )
            if save_hook is not None:
                save_hook(result)
            raise
        finally:
            clear_cuda_cache(args)

    return build_variant_result(
        description=description,
        overrides=overrides,
        runs=runs,
        status="complete",
    )


def build_same_seed_comparison(variant_results: Dict[str, dict], reference_variant: str):
    """Compare variants on the same complete reference seed set."""
    if reference_variant not in variant_results:
        return {}
    reference_seeds = [
        int(r["seed"]) for r in variant_results[reference_variant]["_all_runs"]
    ]
    seed_set = set(reference_seeds)
    out = {
        "reference_variant": reference_variant,
        "seeds": reference_seeds,
        "variants": {},
    }
    for name, item in variant_results.items():
        runs = [r for r in item["_all_runs"] if int(r["seed"]) in seed_set]
        runs = sorted(runs, key=lambda r: reference_seeds.index(int(r["seed"])))
        out["variants"][name] = {
            "summary": exp.summarize_runs(runs),
            "runs": [compact_run(r) for r in runs],
        }
    return out


def strip_private_runs(group_result: dict):
    for shot_result in group_result.values():
        for item in shot_result["variants"].values():
            item.pop("_all_runs", None)


def run_ablation_group(
    base_args,
    group_name: str,
    variants,
    reference_variant: str,
    seeds: List[int],
    seed_map=None,
    group_out=None,
    save_hook=None,
    resume: bool = False,
):
    group_out = group_out if group_out is not None else {}
    for k_shot in base_args.k_shots:
        k_key = f"K={int(k_shot)}"
        shot_seeds = list(seed_map[int(k_shot)]) if seed_map and int(k_shot) in seed_map else seeds
        shot_out = group_out.setdefault(k_key, {})
        shot_out.setdefault("variants", {})
        shot_out["seeds_run"] = shot_seeds
        for variant_name, description, overrides in variants:
            existing_result = shot_out["variants"].get(variant_name)

            def save_variant_progress(result, variant_name=variant_name, shot_out=shot_out):
                shot_out["variants"][variant_name] = result
                if save_hook is not None:
                    save_hook()

            shot_out["variants"][variant_name] = run_variant_for_shot(
                base_args=base_args,
                variant_name=variant_name,
                description=description,
                overrides=overrides,
                k_shot=int(k_shot),
                seeds=shot_seeds,
                existing_result=existing_result,
                save_hook=save_variant_progress,
                resume=resume,
            )
        shot_out["same_seed_comparison_by_reference"] = build_same_seed_comparison(
            shot_out["variants"],
            reference_variant=reference_variant,
        )
        if save_hook is not None:
            save_hook()
    strip_private_runs(group_out)
    return group_out


def filter_readout_variants(cli_args):
    if cli_args.readout_variants is None:
        return READOUT_VARIANTS
    requested = set(cli_args.readout_variants)
    return [item for item in READOUT_VARIANTS if item[0] in requested]


def default_summary_path(cli_args):
    preset = PRESETS[cli_args.preset]
    class_tag = "".join(str(c) for c in preset["class_ids"])
    group = cli_args.ablation
    if group == "q_layers":
        return (
            f"results_{preset['dataset'].lower()}_{class_tag}_qrf_v7_"
            f"ablation_{group}_{len(cli_args.seeds)}seeds.json"
        )
    return f"results_{preset['dataset'].lower()}_{class_tag}_qrf_v7_ablation_{group}_{len(cli_args.seeds)}seeds.json"


def main():
    cli_args = parse_args()
    validate_cli_args(cli_args)
    print(f"RUNNER_VERSION = {RUNNER_VERSION}")
    base_args = apply_common_config(fresh_base_args(), cli_args)
    seeds = list(cli_args.seeds)
    summary_path = Path(cli_args.summary_path or default_summary_path(cli_args))
    seed_map = load_seed_map(cli_args.seed_map_path) if cli_args.seed_map_path else None
    if seed_map:
        missing_k = [int(k) for k in base_args.k_shots if int(k) not in seed_map]
        if missing_k:
            raise ValueError(f"--seed_map_path is missing seeds for k_shots: {missing_k}")
        seeds_config = {f"K={int(k)}": seed_map[int(k)] for k in base_args.k_shots}
    else:
        seeds_config = seeds

    config = {
        "base_script": str(BASE_SCRIPT),
        "preset": cli_args.preset,
        "ablation": cli_args.ablation,
        "dataset": base_args.dataset,
        "class_ids": base_args.class_ids,
        "k_shots": base_args.k_shots,
        "n_qubits": base_args.n_qubits,
        "force_quantum_cpu": base_args.force_quantum_cpu,
        "resume": bool(cli_args.resume),
        "seeds_run": seeds_config,
        "seed_map_path": cli_args.seed_map_path,
        "q_layers_values": cli_args.q_layers_values if cli_args.ablation == "q_layers" else None,
        "readout_observable_counts": {
            name: exp.quantum_readout_dim(base_args.n_qubits, overrides["readout_mode"])
            for name, _, overrides in READOUT_VARIANTS
            if "readout_mode" in overrides
        },
        "readout_note": (
            "All readout variants use Pauli-Z product observables only. "
            "Z, ZZ, ZZZ and ZZZZ are diagonal in the same computational basis, "
            "so higher-order readouts do not require additional measurement bases."
        ),
        "note": (
            "Completed seed runs are saved incrementally and every requested seed is included "
            "in the reported mean/std. No post-hoc seed ranking or metric-based filtering is used. "
            "Use --resume with the same --summary_path to continue after interruption."
        ),
    }

    if cli_args.resume and summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["config"] = config
        payload.setdefault("groups", {})
        print(f"[RESUME] Loaded existing summary: {summary_path}")
    else:
        payload = {
            "config": config,
            "groups": {},
        }
    save_payload(payload, summary_path)

    def save_progress():
        save_payload(payload, summary_path)

    if cli_args.ablation in ("fusion", "all"):
        payload["groups"]["fusion"] = run_ablation_group(
            base_args=base_args,
            group_name="fusion",
            variants=FUSION_VARIANTS,
            reference_variant="gated_logit_residual",
            seeds=seeds,
            seed_map=seed_map,
            group_out=payload["groups"].get("fusion"),
            save_hook=save_progress,
            resume=cli_args.resume,
        )

    if cli_args.ablation in ("quantum", "all"):
        payload["groups"]["quantum"] = run_ablation_group(
            base_args=base_args,
            group_name="quantum",
            variants=QUANTUM_VARIANTS,
            reference_variant="full_reupload_all_pairs",
            seeds=seeds,
            seed_map=seed_map,
            group_out=payload["groups"].get("quantum"),
            save_hook=save_progress,
            resume=cli_args.resume,
        )

    if cli_args.ablation in ("readout", "all"):
        readout_variants = filter_readout_variants(cli_args)
        readout_names = {name for name, _, _ in readout_variants}
        reference_variant = "all_pairs" if "all_pairs" in readout_names else readout_variants[0][0]
        payload["groups"]["readout"] = run_ablation_group(
            base_args=base_args,
            group_name="readout",
            variants=readout_variants,
            reference_variant=reference_variant,
            seeds=seeds,
            seed_map=seed_map,
            group_out=payload["groups"].get("readout"),
            save_hook=save_progress,
            resume=cli_args.resume,
        )

    if cli_args.ablation == "q_layers":
        reference_variant = "q_layers_2" if 2 in cli_args.q_layers_values else f"q_layers_{cli_args.q_layers_values[0]}"
        payload["groups"]["q_layers"] = run_ablation_group(
            base_args=base_args,
            group_name="q_layers",
            variants=build_q_layer_variants(cli_args.q_layers_values),
            reference_variant=reference_variant,
            seeds=seeds,
            seed_map=seed_map,
            group_out=payload["groups"].get("q_layers"),
            save_hook=save_progress,
            resume=cli_args.resume,
        )

    save_payload(payload, summary_path)

    print("\n" + "=" * 80)
    print(f"Saved ablation JSON to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
