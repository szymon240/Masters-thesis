import argparse
import itertools
import json
import os
import platform
import random
import signal
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath("framspy"))

from FramsticksLib import FramsticksLib
from hfc_frams import run_hfc

FRAMS_DIR = "Framsticks54/Framsticks54"
EVALUATIONS_BUDGET = 700000
NUMBER_OF_RUNS = 10
SIM_SETTINGS = "eval-allcriteria-mini.sim;deterministic.sim;sample-period-longest.sim;simulation-2000-steps.sim"

RESULTS_DIR = Path("hfc_grid_results_v2")
CHECKPOINT_DIR = Path("hfc_checkpoints_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

pop_sizes = [200, 500, 1000]
tournament_sizes = [2, 5, 8]
cx_probs = [0.3, 0.6, 0.9]
mut_probs = [0.5, 0.9]
m_values = [5, 10]
r_values = [10, 25]

global_state = {}
global_checkpoint_file = None
global_framsLib = None


def atomic_save_json(data, path: Path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, path)


def load_checkpoint(path: Path):
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            print("Checkpoint uszkodzony, start od nowa.")
    return {
        "combo_id": None,
        "parameters": None,
        "completed_runs": [],
        "run_histories": [],
        "run_final_fits": [],
        "run_best_genotypes": [],
        "run_runtimes": [],
        "run_seeds": [],
        "timestamp_start_iso": None,
        "best_final_fitness": -1.0,
        "best_params": {},
        "best_history": [],
        "best_history_std": []
    }


def save_checkpoint():
    if global_checkpoint_file is not None:
        atomic_save_json(global_state, global_checkpoint_file)


def handle_signal(signum, frame):
    print(f"\nOdebrano sygnał {signum}. Zapisuję checkpoint...")
    try:
        save_checkpoint()
    finally:
        sys.exit(128 + signum)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

def frams_evaluate(individual):
    if individual.fitness is not None:
        return

    try:
        results = global_framsLib.evaluate([f"/*1*/{individual.genotype}"])
    except Exception as e:
        print(f"BŁĄD KRYTYCZNY SYMULATORA: {e}")
        individual.fitness = 0.0
        return

    fit = 0.0

    if results and len(results) > 0:
        first = results[0]
        evals = first.get("evaluations")

        if isinstance(evals, dict):
            # Framsticks często zwraca dane w evals[""]
            nested = evals.get("")
            if isinstance(nested, dict):
                if "velocity" in nested:
                    fit = float(nested["velocity"])
                elif "fit" in nested:
                    fit = float(nested["fit"])
                elif "vertvel" in nested:
                    fit = float(nested["vertvel"])
            else:
                # fallback dla płaskiej struktury
                if "velocity" in evals:
                    fit = float(evals["velocity"])
                elif "fit" in evals:
                    fit = float(evals["fit"])
                elif "vertvel" in evals:
                    fit = float(evals["vertvel"])

    individual.fitness = fit

def make_combo_list():
    return list(itertools.product(
        pop_sizes, tournament_sizes, cx_probs, mut_probs, m_values, r_values
    ))


def aggregate_histories(histories):
    buckets = {}
    for hist in histories:
        for evals, fit in hist:
            buckets.setdefault(evals, []).append(fit)

    eval_points = sorted(buckets.keys())
    avg_history = []
    std_history = []

    for e in eval_points:
        vals = buckets[e]
        avg_history.append([e, float(np.mean(vals))])
        std_history.append([e, float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0])

    return avg_history, std_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-id", type=int, required=True)
    args = parser.parse_args()

    combos = make_combo_list()
    if args.array_id < 0 or args.array_id >= len(combos):
        raise ValueError(f"Invalid array id {args.array_id}, max is {len(combos) - 1}")

    pop, tourn, cxpb, mutpb, m_val, r_val = combos[args.array_id]
    combo_id = f"pop{pop}_t{tourn}_cx{cxpb}_mu{mutpb}_m{m_val}_r{r_val}"

    params = {
        "pop_size": pop,
        "tournament_size": tourn,
        "crossover_prob": cxpb,
        "mutation_prob": mutpb,
        "m_subpopulations": m_val,
        "r_interval": r_val,
    }

    global global_state, global_checkpoint_file, global_framsLib
    global_checkpoint_file = CHECKPOINT_DIR / f"hfc_v2_{combo_id}.json"
    global_state = load_checkpoint(global_checkpoint_file)
    global_state["combo_id"] = combo_id
    global_state["parameters"] = params

    global_framsLib = FramsticksLib(FRAMS_DIR, None, SIM_SETTINGS)

    completed_runs = set(global_state.get("completed_runs", []))
    run_histories = global_state.get("run_histories", [])
    run_final_fits = global_state.get("run_final_fits", [])
    run_best_genotypes = global_state.get("run_best_genotypes", [])
    run_runtimes = global_state.get("run_runtimes", [])
    run_seeds = global_state.get("run_seeds", [])

    base_seed = 300000 + (args.array_id * NUMBER_OF_RUNS)
    start_time = time.time()
    if not global_state.get("timestamp_start_iso"):
        global_state["timestamp_start_iso"] = datetime.now().isoformat()

    for run_id in range(NUMBER_OF_RUNS):
        if run_id in completed_runs:
            continue

        print(f"[{combo_id}] Run {run_id + 1}/{NUMBER_OF_RUNS}")

        seed = base_seed + run_id
        random.seed(seed)
        np.random.seed(seed)

        try:
            run_t0 = time.time()
            history, best_geno = run_hfc(params, EVALUATIONS_BUDGET, frams_evaluate, global_framsLib)
            run_runtime = time.time() - run_t0

            run_histories.append(history)
            run_final_fits.append(history[-1][1] if history else float("-inf"))
            run_best_genotypes.append(best_geno)
            run_runtimes.append(run_runtime)
            run_seeds.append(seed)
            completed_runs.add(run_id)

            avg_h, std_h = aggregate_histories(run_histories)

            global_state["completed_runs"] = sorted(list(completed_runs))
            global_state["run_histories"] = run_histories
            global_state["run_final_fits"] = run_final_fits
            global_state["run_best_genotypes"] = run_best_genotypes
            global_state["run_runtimes"] = run_runtimes
            global_state["run_seeds"] = run_seeds
            global_state["best_final_fitness"] = float(np.max(run_final_fits)) if run_final_fits else -1.0
            global_state["best_params"] = params
            global_state["best_history"] = avg_h
            global_state["best_history_std"] = std_h

            save_checkpoint()

        except Exception as e:
            print(f"[{combo_id}] Run {run_id + 1} failed: {e}")
            traceback.print_exc()
            save_checkpoint()
            break

    if run_histories:
        avg_h, std_h = aggregate_histories(run_histories)

        completed_sorted = global_state["completed_runs"]
        per_run_stats = []
        for idx, rid in enumerate(completed_sorted):
            final_fit = run_final_fits[idx] if idx < len(run_final_fits) else None
            geno = run_best_genotypes[idx] if idx < len(run_best_genotypes) else None
            rt = run_runtimes[idx] if idx < len(run_runtimes) else None
            sd = run_seeds[idx] if idx < len(run_seeds) else None
            per_run_stats.append({
                "run_id": int(rid),
                "seed": int(sd) if sd is not None else None,
                "final_fitness": float(final_fit) if final_fit is not None else None,
                "best_genotype": geno,
                "runtime_seconds": float(rt) if rt is not None else None,
                "status": "completed",
            })

        fits_array = np.array(run_final_fits, dtype=float)
        final_fitness_stats = {
            "min": float(np.min(fits_array)),
            "median": float(np.median(fits_array)),
            "mean": float(np.mean(fits_array)),
            "max": float(np.max(fits_array)),
            "std": float(np.std(fits_array, ddof=1)) if len(fits_array) > 1 else 0.0,
            "q25": float(np.percentile(fits_array, 25)),
            "q75": float(np.percentile(fits_array, 75)),
            "iqr": float(np.percentile(fits_array, 75) - np.percentile(fits_array, 25)),
        }

        champion_idx = int(np.argmax(fits_array))
        champion_overall = {
            "run_id": int(completed_sorted[champion_idx]) if champion_idx < len(completed_sorted) else None,
            "seed": int(run_seeds[champion_idx]) if champion_idx < len(run_seeds) else None,
            "final_fitness": float(fits_array[champion_idx]),
            "genotype": run_best_genotypes[champion_idx] if champion_idx < len(run_best_genotypes) else None,
        }

        result = {
            "algorithm": "HFC",
            "problem": "Velocity",
            "combo_id": combo_id,
            "parameters": params,
            "runs": len(run_histories),
            "completed_runs": global_state["completed_runs"],
            "history": avg_h,
            "history_std": std_h,
            "mean_final_fitness": float(np.mean(run_final_fits)) if run_final_fits else -1.0,
            "std_final_fitness": float(np.std(run_final_fits, ddof=1)) if len(run_final_fits) > 1 else 0.0,
            "elapsed_time": time.time() - start_time,

            "per_run_stats": per_run_stats,
            "final_fitness_stats": final_fitness_stats,
            "champion_overall": champion_overall,

            "metadata": {
                "timestamp_start_iso": global_state.get("timestamp_start_iso"),
                "timestamp_end_iso": datetime.now().isoformat(),
                "evaluations_budget": EVALUATIONS_BUDGET,
                "sim_settings": SIM_SETTINGS,
                "frams_dir": FRAMS_DIR,
                "base_seed": int(base_seed),
                "seeds_used": [int(s) for s in run_seeds],
                "number_of_runs_planned": NUMBER_OF_RUNS,
                "frams_constraints": {
                    "max_numparts": 15,
                    "max_numneurons": 20,
                    "init_iter_max": 100,
                    "init_parts_min": 2,
                    "init_neurons_min": 0,
                },
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
            },
        }

        out_path = RESULTS_DIR / f"{combo_id}.json"
        atomic_save_json(result, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()