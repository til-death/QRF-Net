# -*- coding: utf-8 -*-
"""
FashionMNIST 2/4/6 QRF-Net quantum-depth ablation.

This runner keeps the existing V7 fashion246 configuration fixed and changes
only the number of StronglyEntanglingLayers used inside the quantum circuit.
By default it runs q_layers = 1, 2, 3, 4 over five seeds and saves all runs.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import qrf_net_ablation as ab


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="fashion246", choices=sorted(ab.PRESETS))
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--k_shots", type=int, nargs="*", default=None)
    parser.add_argument("--q_layers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument(
        "--print_params_only",
        action="store_true",
        help="Only print parameter counts for each q_layers value.",
    )
    return parser.parse_args()


def validate_args(cli_args) -> None:
    if not cli_args.seeds:
        raise ValueError("At least one seed is required.")
    if not cli_args.q_layers:
        raise ValueError("At least one q_layers value is required.")
    bad_layers = [x for x in cli_args.q_layers if int(x) < 1]
    if bad_layers:
        raise ValueError(f"q_layers must be >= 1, got: {bad_layers}")


def build_base_args(cli_args):
    base_args = ab.apply_common_config(ab.fresh_base_args(), cli_args)
    base_args.model = "hybrid"
    if cli_args.epochs is not None:
        base_args.epochs = int(cli_args.epochs)
    if cli_args.patience is not None:
        base_args.patience = int(cli_args.patience)
    return base_args


def default_summary_path(cli_args) -> str:
    preset = ab.PRESETS[cli_args.preset]
    class_tag = "".join(str(c) for c in preset["class_ids"])
    seed_tag = f"{len(cli_args.seeds)}seeds"
    return f"results_{preset['dataset'].lower()}_{class_tag}_q_layers_ablation_{seed_tag}.json"


def compact_run(run: dict, q_layers: int) -> dict:
    item = ab.compact_run(run)
    item["q_layers"] = int(q_layers)
    return item


def summarize_layer_runs(runs: List[dict]) -> dict:
    return ab.exp.summarize_runs(runs)


def run_layer_for_shot(base_args, q_layers: int, k_shot: int, seeds: List[int]) -> dict:
    args = copy.deepcopy(base_args)
    args.k_shot = int(k_shot)
    args.q_layers = int(q_layers)

    print("\n" + "=" * 80)
    print(f"[Q-LAYERS ABLATION] q_layers={q_layers} | k={k_shot} | seeds={seeds}")
    print("=" * 80)

    runs = [ab.exp.run_one_seed(args, seed) for seed in seeds]
    return {
        "q_layers": int(q_layers),
        "summary": summarize_layer_runs(runs),
        "runs": [compact_run(run, q_layers=q_layers) for run in runs],
    }


def build_delta_table(layer_results: Dict[str, dict], reference_layers: int = 2) -> Dict[str, dict]:
    ref_key = f"q_layers={int(reference_layers)}"
    if ref_key not in layer_results:
        return {}

    ref_test = layer_results[ref_key]["summary"]["test"]
    out = {
        "reference": ref_key,
        "delta_vs_reference": {},
    }
    for key, item in layer_results.items():
        test = item["summary"]["test"]
        out["delta_vs_reference"][key] = {
            "acc_mean_delta": test["acc_mean"] - ref_test["acc_mean"],
            "f1_mean_delta": test["f1_mean"] - ref_test["f1_mean"],
            "auc_mean_delta": test["auc_mean"] - ref_test["auc_mean"],
        }
    return out


def print_parameter_probe(base_args, q_layers_values: List[int]) -> None:
    print("\nParameter probe")
    print("-" * 48)
    for q_layers in q_layers_values:
        args = copy.deepcopy(base_args)
        args.device = "cpu"
        args.q_layers = int(q_layers)
        model = ab.exp.build_model(args)
        params = ab.exp.count_parameters(model)
        print(f"q_layers={q_layers:<2d} | params={params:,}")


def main():
    cli_args = parse_args()
    validate_args(cli_args)

    base_args = build_base_args(cli_args)
    q_layers_values = [int(x) for x in cli_args.q_layers]
    seeds = [int(x) for x in cli_args.seeds]

    if cli_args.print_params_only:
        print_parameter_probe(base_args, q_layers_values)
        return

    payload = {
        "config": {
            "base_script": str(Path(ab.BASE_SCRIPT).resolve()),
            "preset": cli_args.preset,
            "dataset": base_args.dataset,
            "class_ids": base_args.class_ids,
            "k_shots": base_args.k_shots,
            "q_layers": q_layers_values,
            "seeds": seeds,
            "note": "Only q_layers is varied; all five configured seeds are saved.",
            "fixed_quantum_config": {
                "n_qubits": base_args.n_qubits,
                "reuploads": base_args.reuploads,
                "readout_mode": base_args.readout_mode,
                "fusion_mode": base_args.fusion_mode,
                "hybrid_style": base_args.hybrid_style,
            },
        },
        "shots": {},
    }

    for k_shot in base_args.k_shots:
        k_key = f"K={int(k_shot)}"
        payload["shots"][k_key] = {
            "layers": {},
        }
        for q_layers in q_layers_values:
            layer_key = f"q_layers={int(q_layers)}"
            payload["shots"][k_key]["layers"][layer_key] = run_layer_for_shot(
                base_args=base_args,
                q_layers=int(q_layers),
                k_shot=int(k_shot),
                seeds=seeds,
            )
        payload["shots"][k_key]["delta_table"] = build_delta_table(
            payload["shots"][k_key]["layers"],
            reference_layers=2,
        )

    summary_path = cli_args.summary_path or default_summary_path(cli_args)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("[Q-LAYERS ABLATION SUMMARY]")
    for k_key, shot_result in payload["shots"].items():
        print(f"\n{k_key}")
        for layer_key, item in shot_result["layers"].items():
            test = item["summary"]["test"]
            params = item["summary"]["params"]
            print(
                f"{layer_key:12s} | Params={params:,} | "
                f"Acc={test['acc_mean']:.4f}+/-{test['acc_std']:.4f} | "
                f"F1={test['f1_mean']:.4f}+/-{test['f1_std']:.4f} | "
                f"AUC={test['auc_mean']:.4f}+/-{test['auc_std']:.4f}"
            )
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
