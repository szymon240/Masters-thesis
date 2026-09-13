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
from fuds_frams import run_fuds

FRAMS_DIR = "Framsticks54/Framsticks54"
EVALUATIONS_BUDGET = 700000
NUMBER_OF_RUNS = 10
MID_CHECKPOINT_INTERVAL = 150000  # Co ile ewaluacji emitowac mid-run checkpoint (150k = ~1-2h)
SIM_SETTINGS = "eval-allcriteria-mini.sim;deterministic.sim;sample-period-longest.sim;simulation-2000-steps.sim"

RESULTS_DIR = Path("fuds_grid_results_v2")
CHECKPOINT_DIR = Path("fuds_checkpoints_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

pop_sizes = [200, 500, 1000]
tournament_sizes = [2, 5, 8]
cx_probs = [0.3, 0.6, 0.9]
mut_probs = [0.5, 0.9]

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
        "timestamp_start_iso": None,
        "per_run_data": {},
        "per_run_mid_state": {},
        "per_run_completed_budget": {},
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
        pop_sizes, tournament_sizes, cx_probs, mut_probs
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

    pop, tourn, cxpb, mutpb = combos[args.array_id]
    combo_id = f"pop{pop}_t{tourn}_cx{cxpb}_mu{mutpb}"

    params = {
        "pop_size": pop,
        "tournament_size": tourn,
        "crossover_prob": cxpb,
        "mutation_prob": mutpb,
    }

    global global_state, global_checkpoint_file, global_framsLib
    global_checkpoint_file = CHECKPOINT_DIR / f"fuds_v2_{combo_id}.json"
    global_state = load_checkpoint(global_checkpoint_file)
    global_state["combo_id"] = combo_id
    global_state["parameters"] = params

    # Inicjalizacja brakujacych pol w starych checkpointach (zachowanie kompatybilnosci)
    global_state.setdefault("per_run_data", {})
    global_state.setdefault("per_run_mid_state", {})
    global_state.setdefault("per_run_completed_budget", {})

    global_framsLib = FramsticksLib(FRAMS_DIR, None, SIM_SETTINGS)

    per_run_data = global_state["per_run_data"]
    per_run_mid_state = global_state["per_run_mid_state"]
    per_run_completed_budget = global_state["per_run_completed_budget"]

    # Seedy: paczka 10 unikalnych dla kazdej kombinacji, offset algorytmu = 600000 (FUDS)
    # Zakres: 600000 .. 600539 (54 kombinacji x 10 runow). Brak nakladek z innymi algorytmami.
    base_seed = 600000 + (args.array_id * NUMBER_OF_RUNS)
    start_time = time.time()
    if not global_state.get("timestamp_start_iso"):
        global_state["timestamp_start_iso"] = datetime.now().isoformat()

    for run_id in range(NUMBER_OF_RUNS):
        rid_key = str(run_id)
        completed_budget = int(per_run_completed_budget.get(rid_key, 0))

        # Skip jezeli juz ukonczone na obecnym lub wyzszym budzecie
        if completed_budget >= EVALUATIONS_BUDGET:
            continue

        resume_state = per_run_mid_state.get(rid_key)
        seed = base_seed + run_id

        if resume_state is None:
            random.seed(seed)
            np.random.seed(seed)
            print(f"[{combo_id}] Run {run_id + 1}/{NUMBER_OF_RUNS} (fresh, seed={seed})")
        else:
            print(
                f"[{combo_id}] Run {run_id + 1}/{NUMBER_OF_RUNS} "
                f"(resume from evals={resume_state.get('evaluations', 0)}, "
                f"budget {completed_budget}->{EVALUATIONS_BUDGET})"
            )

        # Closure capturing rid_key for callback
        def make_callback(this_rid_key):
            def cb(state):
                global_state["per_run_mid_state"][this_rid_key] = state
                save_checkpoint()
            return cb

        try:
            run_t0 = time.time()
            history, best_geno = run_fuds(
                params,
                EVALUATIONS_BUDGET,
                frams_evaluate,
                global_framsLib,
                resume_state=resume_state,
                mid_checkpoint_callback=make_callback(rid_key),
                mid_checkpoint_interval=MID_CHECKPOINT_INTERVAL,
            )
            run_runtime = time.time() - run_t0

            # Cumulative runtime: jezeli to extension, dodaj do poprzedniego czasu
            prev_data = per_run_data.get(rid_key, {})
            prev_runtime = float(prev_data.get("runtime_seconds") or 0.0)

            per_run_data[rid_key] = {
                "run_id": run_id,
                "seed": int(seed),
                "final_fitness": float(history[-1][1]) if history else float("-inf"),
                "best_genotype": best_geno,
                "runtime_seconds": float(prev_runtime + run_runtime),
                "history": history,
                "status": "completed",
            }
            per_run_completed_budget[rid_key] = EVALUATIONS_BUDGET

            # Auto-cleanup: usun mid_state dla ukonczonego runa (nie potrzebny dla resume).
            # Zapobiega akumulacji ogromnych plikow checkpoint.
            if rid_key in per_run_mid_state:
                del per_run_mid_state[rid_key]

            save_checkpoint()

        except Exception as e:
            print(f"[{combo_id}] Run {run_id + 1} failed: {e}")
            traceback.print_exc()
            save_checkpoint()
            break

    # Zbudowanie finalnego wyniku z per_run_data (sortowane wg run_id)
    sorted_rids = sorted(per_run_data.keys(), key=int)
    if sorted_rids:
        run_histories = [per_run_data[k]["history"] for k in sorted_rids]
        run_final_fits = [per_run_data[k]["final_fitness"] for k in sorted_rids]
        run_best_genotypes = [per_run_data[k]["best_genotype"] for k in sorted_rids]
        run_runtimes = [per_run_data[k]["runtime_seconds"] for k in sorted_rids]
        run_seeds = [per_run_data[k]["seed"] for k in sorted_rids]
        completed_run_ids = [int(k) for k in sorted_rids]

        avg_h, std_h = aggregate_histories(run_histories)

        per_run_stats = [{
            "run_id": per_run_data[k]["run_id"],
            "seed": per_run_data[k]["seed"],
            "final_fitness": per_run_data[k]["final_fitness"],
            "best_genotype": per_run_data[k]["best_genotype"],
            "runtime_seconds": per_run_data[k]["runtime_seconds"],
            "status": per_run_data[k]["status"],
        } for k in sorted_rids]

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

        # MAKSYMALIZACJA: champion = run z NAJWIEKSZYM final fitness
        champion_idx = int(np.argmax(fits_array))
        champion_overall = {
            "run_id": int(completed_run_ids[champion_idx]),
            "seed": int(run_seeds[champion_idx]),
            "final_fitness": float(fits_array[champion_idx]),
            "genotype": run_best_genotypes[champion_idx],
        }

        result = {
            "algorithm": "FUDS",
            "problem": "Velocity",
            "combo_id": combo_id,
            "parameters": params,
            "runs": len(run_histories),
            "completed_runs": completed_run_ids,
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
                "mid_checkpoint_interval": MID_CHECKPOINT_INTERVAL,
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