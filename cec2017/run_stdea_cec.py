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

# Wymaga instalacji środowiska CEC2017
from cec2017.cec2017 import functions as cec_functions
import config
from stdea import run_stdea

RESULTS_DIR = Path("stdea_cec_results_v2")
CHECKPOINT_DIR = Path("stdea_cec_checkpoints_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

global_state = {}
global_checkpoint_file = None
curr_func = None
curr_optimum = None

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
        "combo_id": None, "parameters": None, "completed_runs": [],
        "run_histories": [], "run_final_fits": [],
        "run_best_solutions": [], "run_runtimes": [], "run_seeds": [],
        "timestamp_start_iso": None,
        "best_final_fitness": float('inf'),
        "best_params": {}, "best_history": [], "best_history_std": []
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

def wrapper(individual):
    array = np.array([individual.genes])
    try:
        val = curr_func(array)[0]
    except Exception as e:
        print(f"Błąd krytyczny CEC: {e}")
        val = float('inf')
    # Normalizacja błędu do 0
    individual.fitness = val - curr_optimum

def make_combo_list():
    funcs = config.FUNCTIONS_IDS
    pops = config.EXPERIMENTAL_PARAMS["pop_size"]
    tourns = config.EXPERIMENTAL_PARAMS["tournament_size"]
    cxs = config.EXPERIMENTAL_PARAMS["crossover_prob"]
    muts = config.EXPERIMENTAL_PARAMS["mutation_prob"]
    alphas = config.EXPERIMENTAL_PARAMS["alpha"]
    return list(itertools.product(funcs, pops, tourns, cxs, muts, alphas))

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

    f_id, pop, tourn, cxpb, mutpb, alpha_val = combos[args.array_id]
    combo_id = f"F{f_id}_pop{pop}_t{tourn}_cx{cxpb}_mu{mutpb}_a{alpha_val}"

    params = {
        "function_id": f_id,
        "pop_size": pop,
        "tournament_size": tourn,
        "crossover_prob": cxpb,
        "mutation_prob": mutpb,
        "alpha": alpha_val
    }

    global global_state, global_checkpoint_file, curr_func, curr_optimum

    curr_func = getattr(cec_functions, f"f{f_id}")
    curr_optimum = f_id * 100.0

    global_checkpoint_file = CHECKPOINT_DIR / f"stdea_v2_{combo_id}.json"
    global_state = load_checkpoint(global_checkpoint_file)
    global_state["combo_id"] = combo_id
    global_state["parameters"] = params

    completed_runs = set(global_state.get("completed_runs", []))
    run_histories = global_state.get("run_histories", [])
    run_final_fits = global_state.get("run_final_fits", [])
    run_best_solutions = global_state.get("run_best_solutions", [])
    run_runtimes = global_state.get("run_runtimes", [])
    run_seeds = global_state.get("run_seeds", [])

    base_seed = 600000 + args.array_id * config.RUNS
    start_time = time.time()
    if not global_state.get("timestamp_start_iso"):
        global_state["timestamp_start_iso"] = datetime.now().isoformat()

    for run_id in range(config.RUNS):
        if run_id in completed_runs:
            continue

        print(f"[{combo_id}] Run {run_id + 1}/{config.RUNS}")

        seed = base_seed + run_id
        random.seed(seed)
        np.random.seed(seed)

        try:
            run_t0 = time.time()
            history, best_sol = run_stdea(params, config.EVALUATIONS_BUDGET, wrapper, config.D, config.MIN_VAL, config.MAX_VAL)
            run_runtime = time.time() - run_t0

            run_histories.append(history)
            run_final_fits.append(history[-1][1] if history else float("inf"))
            run_best_solutions.append(best_sol)
            run_runtimes.append(run_runtime)
            run_seeds.append(seed)
            completed_runs.add(run_id)

            avg_h, std_h = aggregate_histories(run_histories)

            global_state["completed_runs"] = sorted(list(completed_runs))
            global_state["run_histories"] = run_histories
            global_state["run_final_fits"] = run_final_fits
            global_state["run_best_solutions"] = run_best_solutions
            global_state["run_runtimes"] = run_runtimes
            global_state["run_seeds"] = run_seeds
            global_state["best_final_fitness"] = float(np.min(run_final_fits)) if run_final_fits else float('inf')
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

        # --- WARSTWA B: szczegoly per-run ---
        completed_sorted = global_state["completed_runs"]
        per_run_stats = []
        for idx, rid in enumerate(completed_sorted):
            final_fit = run_final_fits[idx] if idx < len(run_final_fits) else None
            sol = run_best_solutions[idx] if idx < len(run_best_solutions) else None
            rt = run_runtimes[idx] if idx < len(run_runtimes) else None
            sd = run_seeds[idx] if idx < len(run_seeds) else None
            per_run_stats.append({
                "run_id": int(rid),
                "seed": int(sd) if sd is not None else None,
                "final_fitness": float(final_fit) if final_fit is not None else None,
                "best_solution": sol,
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

        champion_idx = int(np.argmin(fits_array))
        champion_overall = {
            "run_id": int(completed_sorted[champion_idx]) if champion_idx < len(completed_sorted) else None,
            "seed": int(run_seeds[champion_idx]) if champion_idx < len(run_seeds) else None,
            "final_fitness": float(fits_array[champion_idx]),
            "solution": run_best_solutions[champion_idx] if champion_idx < len(run_best_solutions) else None,
        }

        result = {
            "algorithm": "StdEA",
            "problem": "CEC2017",
            "combo_id": combo_id,
            "parameters": params,
            "runs": len(run_histories),
            "completed_runs": global_state["completed_runs"],
            "history": avg_h,
            "history_std": std_h,
            "mean_final_fitness": float(np.mean(run_final_fits)) if run_final_fits else float('inf'),
            "std_final_fitness": float(np.std(run_final_fits, ddof=1)) if len(run_final_fits) > 1 else 0.0,
            "elapsed_time": time.time() - start_time,

            # --- WARSTWA B ---
            "per_run_stats": per_run_stats,
            "final_fitness_stats": final_fitness_stats,
            "champion_overall": champion_overall,

            # --- WARSTWA A: metadata reprodukowalnosci ---
            "metadata": {
                "timestamp_start_iso": global_state.get("timestamp_start_iso"),
                "timestamp_end_iso": datetime.now().isoformat(),
                "evaluations_budget": config.EVALUATIONS_BUDGET,
                "dim": config.D,
                "min_val": config.MIN_VAL,
                "max_val": config.MAX_VAL,
                "function_id": f_id,
                "function_optimum": curr_optimum,
                "base_seed": int(base_seed),
                "seeds_used": [int(s) for s in run_seeds],
                "number_of_runs_planned": config.RUNS,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
            },
        }

        out_path = RESULTS_DIR / f"{combo_id}.json"
        atomic_save_json(result, out_path)
        print(f"Zapisano w pełni ukończone wyniki: {out_path}")

if __name__ == "__main__":
    main()